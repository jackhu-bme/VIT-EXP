from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score


import torch
from torch import nn

from data import create_train_dl_list, create_valid_dl_list

import numpy as np
import pandas as pd
# import tqdm
from tqdm import tqdm


from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP
import os

# import itertools

import random

# import wandb

# helpers
def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array




def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data


# class CycleSampler:
#     def __init__(self, lengths, ratios):
#         """
#         lengths: 每个数据集的长度列表
#         ratios: 采样比例列表
#         """
#         self.lengths = lengths
#         self.ratios = ratios
#         self.ratio_sum = sum(ratios)
#         self.total_length = sum(lengths)
#         self.indices = []

#         # 根据比例构造初始索引
#         start = 0
#         start_list = []
#         datasets = []
#         for i in range(len(lengths)):
#             start_list.append(start)
#             datasets.append(list(range(start, start + lengths[i])))
#             start += lengths[i]
            
        
#         self.datasets = datasets

#         self.start_list = start_list
        
#         self.chances = [ratio / self.ratio_sum for ratio in self.ratios]

#     def __iter__(self):
#         while True:
#             random_num = random.random()
#             # choose the dataset according to the chances
#             dataset_index = 0
#             for i, chance in enumerate(self.chances):
#                 if random_num < chance:
#                     dataset_index = i
#                     break
#                 random_num -= chance
#             # choose the sample from the dataset
#             start = self.start_list[dataset_index]
#             length = self.lengths[dataset_index]
#             index = random.randint(start, start + length - 1)
#             yield index

#     def __len__(self):
#         return len(self.indices)




def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

def create_accelerate_kwargs(config):
    gradient_accumulation_steps = config["trainer"].get("gradient_accumulation_steps", 1)
    accelerate_kwargs = {
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    return accelerate_kwargs

class RandDatasetSampler():
    def __init__(self, config):
        self.ratio_list = config["ratio_list"]
        assert sum(self.ratio_list) > 0, "the sum of ratio list should be a positive number"
        # norm the ratio list
        self.ratio_list = [ratio / sum(self.ratio_list) for ratio in self.ratio_list]
        self.n_datasets = len(self.ratio_list)
    
    def sample(self, step):
        # generate a random number according to step
        random_num = random.random()
        # choose the dataset according to the chances
        dataset_index = 0
        for i, chance in enumerate(self.ratio_list):
            if random_num < chance:
                dataset_index = i
                break
            random_num -= chance
        acc_steps_list = [0, ] * self.n_datasets
        acc_steps_list[dataset_index] = 1
        return acc_steps_list

class CombinedDatasetSampler():
    """
    CombinedDatasetSampler is a sampler that combines multiple dataset samplers, defines the number of steps to accumulate the gradients for each dataset
    """
    def __init__(self, config):
        self.acc_steps_list = config["acc_steps_list"]
        assert sum(self.acc_steps_list) > 0, "the sum of acc_steps_list should be a positive number"
        # assert each element in acc_steps_list is a positive int
        self.acc_steps_list = [int(acc_steps) for acc_steps in self.acc_steps_list]
        assert all([acc_steps >= 0 for acc_steps in self.acc_steps_list]), "each element in acc_steps_list should be a non-negative int"
        self.n_datasets = len(self.acc_steps_list)
    
    def sample(self, step):
        # generate a random number according to step
        return self.acc_steps_list


class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        tokenizer=None,
        config=None,
        # num_train_steps,
        # batch_size,
        # data_train = "train",
        # data_valid = "valid",
        # use_seg = False,
        # seg_data_train = None,
        # seg_data_valid = None,
        # seg_mask_train = None,
        # seg_mask_valid = None,
        # balance_report_seg = 1.0,
        # reports_file_train = "data_reports.xslx",
        # reports_file_valid = "data_reports.xslx",
        # labels = "labels.csv",
        # tokenizer = None,
        # lr = 1.25e-6,
        # wd = 0.,
        # max_grad_norm = 0.5,
        # save_results_every = 1000,
        # save_model_every = 1000 ,
        results_folder = '/shares/menze.dqbm.uzh/ihamam/ctclip/',
        # num_workers = 8,
        # accelerate_kwargs: dict = dict(),
        resume_path = None,
        auto_resume = False,
        # metadata_train = "train_metadata.csv",
        wandb_logger = None,
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerate_kwargs = create_accelerate_kwargs(config)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            self.tokenizer=BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

        self.register_buffer('steps', torch.Tensor([0]))

        trainer_config = config["trainer"]

        self.num_train_steps = trainer_config["num_train_steps"]
        # self.batch_size = batch_size # the batch size is defined for different datasets individually, so no need to define it here

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=trainer_config["lr"], wd=trainer_config["wd"])

        self.max_grad_norm = trainer_config["max_grad_norm"]
        
        self.lr=trainer_config["lr"]

        self.dl_list = create_train_dl_list(config["train_data_list"])
        # print(f"datasets: {self.dl_list}")

        self.valid_dl_list = create_valid_dl_list(config["valid_data_list"])

        self.dataset_sampler = self.create_dataset_sampler(config["DatasetSampler"])

        # if not use_seg:
        #     self.ds = CTReportDataset(data_folder=data_train, csv_file=reports_file_train, metadata_train=metadata_train)
        #     self.valid_ds = CTReportDatasetinfer(data_folder=data_valid, csv_file=reports_file_valid, labels = labels)
        # else:
        #     self.ds = CTReportSegDataset(data_folder=data_train, csv_file=reports_file_train, metadata_train=metadata_train, 
        #                                  seg_data=seg_data_train, seg_mask=seg_mask_train)
        #     self.valid_ds = CTReportDatasetinfer(data_folder=data_valid, csv_file=reports_file_valid, labels = labels)
            # todo: add the seg support in valid_ds
        
        self.balance_loss_weight = trainer_config.get("balance_loss_weight", [1.0, ] * len(self.dl_list))

        # prepare with accelerator
        # self.dl_iter_list = [cycle(dl) for dl in self.dl_list]
        # self.valid_dl_iter_list = [cycle(valid_dl) for valid_dl in self.valid_dl_list]
        # self.dl_iter=cycle(self.dl)
        # self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        # (
 		# 	self.dl_iter,
        #     self.valid_dl_iter,
        #     self.CTClip,
        #     self.optim,
        # ) = self.accelerator.prepare(
        #     self.dl_iter,
        #     self.valid_dl_iter,
        #     self.CTClip,
        #     self.optim,
        # )

        self.dl_iter_list = [cycle(self.accelerator.prepare_data_loader(dl)) for dl in self.dl_list]
        self.dl_step_list = [0, ] * len(self.dl_list)
        self.valid_dl_iter_list = [cycle(self.accelerator.prepare_data_loader(valid_dl)) for valid_dl in self.valid_dl_list]
        self.valid_dl_step_list = [0, ] * len(self.valid_dl_list)
        # self.CTClip = self.accelerator.prepare_model(self.CTClip)
        # self.optim = self.accelerator.prepare_optimizer(self.optim)\
        # in future, if use scheduler
        # fake a scheduler
        
        # self.scheduler = CosineAnnealingWarmUpRestarts(self.optim,
        #                                                 T_0=4000000,    # Maximum number of iterations
        #                                                 T_warmup=1000, # Number of warmup steps
        #                                                 eta_max=self.lr)
        
        # self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=1000, gamma=1.0)
        (self.CTClip, self.optim, self.scheduler) = self.accelerator.prepare(
            self.CTClip, self.optim, self.scheduler
        )
        
        # register for checkpointing
        self.accelerator.register_for_checkpointing(self.scheduler)
           

        self.save_model_every = trainer_config["save_model_every"]
        self.save_results_every = trainer_config["save_results_every"]

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0: #and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            # rmtree(str(self.results_folder))
            print(f"detecting previous experiment checkpoints and results, resume from them by default")
            print(f"results folder: {self.results_folder}")

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.auto_resume = auto_resume

        if resume_path is not None:
            # self.load_model(resume_path)
            self.print(f"resuming the sheduler and the model from {resume_path}")
            # set the step according to the model's name
            self.print(f"before loading, steps: {self.steps}")
            # self.resume_step = int(os.path.basename(resume_path).split(".")[-2]) # this is for old ctclip version checkpoints
            self.resume_step = int(resume_path.split("_")[1].split(".")[0])
            self.steps += self.resume_step
            self.print(f"resuming from step {self.steps} according to the model's name: {resume_path}")
            self.load(resume_path)
            # restore the state of the dataloader
            self.dl = accelerate.skip_first_batches(self.dl, self.steps)
        elif auto_resume:
            # try to find the lastest checkpoint that is saved properly and could be loaded
            ckpt_list = sorted([*self.results_folder.glob('checkpoint_*')])
            # sort the ckpt_list according to the step
            ckpt_list = sorted(ckpt_list, key=lambda x: int(x.name.split("_")[1].split(".")[0]))
            chosen_ckpt = None
            for ckpt in ckpt_list:
                try:
                    self.print(f"try to load from checkpoint: {ckpt}")
                    self.load(ckpt)
                    self.print(f"successfully loaded from checkpoint: {ckpt}")
                    chosen_ckpt = ckpt
                    break
                except Exception as e:
                    self.print(f"failed to load from checkpoint: {ckpt}, error: {e}")
                    continue
            if chosen_ckpt is not None:
                self.print(f"resuming from step {self.steps} according to the model's name: {chosen_ckpt}")
                self.resume_step = int(ckpt.name.split("_")[1].split(".")[0])
                self.steps += self.resume_step
                self.dl = accelerate.skip_first_batches(self.dl, self.steps)
            else:
                self.resume_step = None
                print(f"no valid checkpoint found for auto resume, start from scratch")
                # raise ValueError("no valid checkpoint found for auto resume")
        else:
            self.resume_step = None
        
        self.wandb_logger = wandb_logger
        

    def create_dataset_sampler(self, config):
        sampler_type = config["type"]
        if sampler_type == "Random":
            return RandDatasetSampler(config)
        elif sampler_type == "Combined":
            return CombinedDatasetSampler(config)
        else:
            raise ValueError(f"DatasetSampler type {sampler_type} is not supported")
        


    # def save(self, path):
    #     # Ensure that the required attributes are not None before attempting to save
    #     if self.accelerator is None or self.CTClip is None or self.optim is None:
    #         raise ValueError("Accelerator, CTClip model, or optimizer is not initialized.")

    #     if not self.accelerator.is_local_main_process:
    #         return

    #     pkg = dict(
    #         model=self.accelerator.get_state_dict(self.CTClip),
    #         optim=self.optim.state_dict(),
    #     )
    #     torch.save(pkg, path)

    # def load_model(self, path):
    #     path = Path(path)
    #     assert path.exists()
    #     pkg = torch.load(path)

    #     CTClip = self.accelerator.unwrap_model(self.CTClip)
    #     try:
    #         CTClip.load_state_dict(pkg)
    #     except Exception as e:
    #         print(f"try removing the module prefix from the model state dict")
    #         new_state_dict = {}
    #         for k, v in pkg.items():
    #             if k.startswith('module.'):
    #                 k = k[7:]
    #             new_state_dict[k] = v
    #         CTClip.load_state_dict(new_state_dict)
    #         print(f"successfully loaded the model state dict after removing the module prefix")

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.accelerator.load_state(path)
        # pkg = torch.load(path)

        # CTClip = self.accelerator.unwrap_model(self.CTClip)
        # CTClip.load_state_dict(pkg['model'])

        # self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def prepare_batch(self, batch):
        """
        make essential data preprocess on gpu, such as tokenization, and move the data to gpu
        """
        if batch["data_type"][0] == "imagereport":
            text = batch['text']
            video = batch['image']
            video = video.to(self.device)
            text = list(text)
            text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)
            batch["text"] = text_tokens
            batch["image"] = video
        elif batch["data_type"][0] == "imageseg" or batch["data_type"][0] == "imagesegreport":
            seg_mask = batch["seg_mask"].to(self.device)
            seg_data = batch["image"].to(self.device)
            batch["seg_mask"] = seg_mask
            batch["image"] = seg_data
        else:
            raise ValueError(f"unsupported data type: {batch['data_type']}")
        return batch


    def train_step_single_dataset(self, dataset_index=None):
        batch = next(self.dl_iter_list[dataset_index])
        batch = self.prepare_batch(batch)
        #video = video
        with self.accelerator.accumulate(self.CTClip):
            with self.accelerator.autocast():
                loss, loss_dict = self.CTClip(batch, return_loss=True, return_loss_dict=True, 
                                              device=self.device, accelerator=self.accelerator)
                # times the weight for this dataset to the loss and loss dict
                loss = loss * self.balance_loss_weight[dataset_index]
                bal_loss_dict = {}
                for key, value in loss_dict.items():
                    bal_loss_dict[key] = value * self.balance_loss_weight[dataset_index]
        self.accelerator.backward(loss)
        return bal_loss_dict

    @staticmethod
    def loss_update(loss_dict, loss_dict_single):
        for key, value in loss_dict_single.items():
            loss_dict[key] = loss_dict.get(key, 0) + value
        return loss_dict


    def train_step_single(self):
        """
        in single training step, we deal with multiple datasets, how to train them in a single step and accumulate the loss?
        this behavior is defined by our DatasetSampler
        one possible logic of dataset sampler is to sample only a random type by the defined ratio of different datasets
        another possible logic is to sample from all datasets in a single step, and number of accumulated steps is defined by the DatasetSampler
        however, they can be unified into one type of sampler, the sampler takes the input of step, output a list [n1, n2, n3, n4, ...] to determine the number of steps to accumulate the gradients for each dataset
        the batch size used for each dataset is defined by the dataloader, so we don't need to worry about the batch size
        """
        acc_steps_list = self.dataset_sampler.sample(self.steps.item())
        loss_dict = {}
        for i, acc_step in enumerate(acc_steps_list):
            for _ in range(acc_step):
                loss_dict_single = self.train_step_single_dataset(dataset_index=i)
                loss_dict = self.loss_update(loss_dict, loss_dict_single)
        return loss_dict


    def train_step(self):
        steps = int(self.steps.item())

        self.CTClip.train()
        # logs
        logs = {}
        # update CTClip model
        # video, text = next(self.dl_iter)
        loss_dict = self.train_step_single()
        
        to_acc_dict = loss_dict.copy()
        to_acc_dict["step"] = self.steps.item()
        accum_log(logs, to_acc_dict)
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        # self.print(f"{steps}: loss: {logs['loss']}")
        self.print(f"log: {logs}")

        self.wandb_logger.log(logs, step=self.steps.int().item())

        # if self.is_main and not (steps % self.save_results_every):
        #     with torch.no_grad():

        #         models_to_evaluate = ((self.CTClip, str(steps)),)

        #         print(f"evaluating model: {steps}")

        #         for model, filename in models_to_evaluate:
        #             model.eval()
        #             predictedall=[]
        #             realall=[]

        #             #Fast inference on 100 images
        #             for i in range(10):
        #                 print(f"test i: {i}")
        #                 valid_data, text, onehotlabels, name_acc = next(self.valid_dl_iter)
        #                 valid_data = valid_data.to(device)

        #                 if "module" in model.__dict__:
        #                     model = model.module

        #                 pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
        #                 plotdir = str(self.results_folder / f'CTClip_{steps}' )
        #                 plotdir = plotdir + "/"

        #                 Path(plotdir).mkdir(parents=True, exist_ok=True)

        #                 predictedlabels=[]
        #                 for pathology in pathologies:
        #                     text = [f"There is {pathology}.", f"There is no {pathology}."]
        #                     text_tokens=self.tokenizer(
        #                                     text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        #                     output = model(text_tokens, valid_data,  device=device)


        #                     output = apply_softmax(output)

        #                     print(output)
        #                     append_out=output.detach().cpu().numpy()
        #                     print(output)
        #                     if output[0]>output[1]:
        #                         predictedlabels.append(append_out[0])
        #                     else:
        #                         predictedlabels.append(append_out[0])
        #                 predictedall.append(predictedlabels)
        #                 realall.append(onehotlabels.detach().cpu().numpy()[0])
        #                 # Print and save classification report
        #             realall=np.array(realall)
        #             predictedall=np.array(predictedall)

        #             dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)
        #             realall = np.rint(realall).astype(int)
        #             predictedall = np.rint(predictedall).astype(int)


        #             print('Test F1 Accuracy: ', f1_score(realall, predictedall,average='micro'))
        #             print('Test Flat Accuracy: ', accuracy_score(realall.flatten(), predictedall.flatten()),'\n')

        #             writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

        #             dfs.to_excel(writer, sheet_name='Sheet1', index=False)

        #             writer.close()
        #             del output


        if not (steps % self.save_model_every):
            # state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
            # the following code will also work, and only get state_dict on rank0
            # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # with FSDP.state_dict_type(self.CTCLIP, StateDictType.FULL_STATE_DICT, save_policy):
            #     state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
            if self.is_main:
                print(f"save model at step: {steps}, output_dir: {self.results_folder}")
                self.accelerator.save_state(output_dir=self.results_folder)
                # model_path = str(self.results_folder / f'CTClip.{steps}.pt')
                # self.accelerator.save(state_dict, model_path)
        self.steps += 1
        return logs


    def train(self, log_fn=noop):
        # device = next(self.CTClip.parameters()).device
        # device = torch.device('cuda')
        with tqdm(total=self.num_train_steps, desc='Training', unit='step') as pbar:
            if self.resume_step is not None:
                pbar.update(self.resume_step)
            while self.steps < self.num_train_steps:
                logs = self.train_step()
                log_fn(logs)
                pbar.update(1)
