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

from zero_shot import ctclip_image_report_zero_shot_cls_test

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

import wandb




def radgenome_image_open_seg_test_ten_images(model):
    # create radgenome validation image dataset and dataloader
    data_config_dict = {}
    data_config_dict["seg_data_valid"] = "/mnt/input/RadGenome/valid_preprocessed_img_organized_fp16"
    data_config_dict["metadata_valid"] = "/mnt/input/CT-RATE/organized_dataset/csv_dir/metadata/validation_metadata.csv"
    data_config_dict["seg_mask_valid"] = "/mnt/input/RadGenome/valid_preprocessed_mask_combined_bool"
    data_config_dict["seg_mask_name_table"] = "/mnt/input/RadGenome/label_mappings/radgenome_labels.xlsx"
    data_config_dict["seg_mask_prompt_type"] = "this_is"
    data_config_dict["type"] = "imageopenseg"
    data_config_dict["batch_size"] = 1
    data_config_dict["num_workers"] = 0

    valid_dl = create_valid_dl_list([data_config_dict])[0]

    all_vis_dict = {}

    # visualize the segmentation results, for each image in the batch
    # 10 images for most

    device = torch.device('cuda:0')

    for i, batch in enumerate(valid_dl):
        if i >= 10:
            break
        seg_mask = batch["seg_mask"].to(device)
        seg_data = batch["image"].to(device)
        batch["seg_mask"] = seg_mask
        batch["image"] = seg_data
        seg_mask_promp_dict = batch["seg_mask_promp_dict"]
        for key, value in seg_mask_promp_dict.items():
            seg_mask_promp_dict[key] = value.to(device)
        batch["seg_mask_promp_dict"] = seg_mask_promp_dict
        _, _, vis_res = model(batch, return_vis=True, img_prefix=f"valid_{i}")
        all_vis_dict.update(vis_res)

    return {"to_vis_dict": all_vis_dict}
    


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

def create_valid_tests(test_name_list):
    tests = []
    for test_name in test_name_list:
        if test_name == "ctclip_image_report_zero_shot_cls_test":
            tests.append(ctclip_image_report_zero_shot_cls_test)
        elif test_name == "radgenome_image_open_seg_test_ten_images":
            tests.append(radgenome_image_open_seg_test_ten_images)
        else:
            raise ValueError(f"test name {test_name} is not supported")
    return tests


class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        tokenizer=None,
        accelerator_kwargs=None,
        wandb_init_kwargs=None,
        config=None,
        results_folder = '/shares/menze.dqbm.uzh/ihamam/ctclip/',
        resume_path = None,
        auto_resume = False,
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
        if accelerator_kwargs is None:
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerator_kwargs)
        else:
            print(f"accelerator kwargs: {accelerator_kwargs}")
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerator_kwargs)

        # init the wandb tracker
        if wandb_init_kwargs is not None:
            self.accelerator.init_trackers(**wandb_init_kwargs)

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

        # self.valid_dl_list = create_valid_dl_list(config["valid_data_list"])

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

        self.valid_tests = create_valid_tests(config["valid_test_list"])

        self.vis_val_tests = create_valid_tests(config.get("sample_test_list", []))

        self.vis_train_interval = trainer_config.get("vis_train_every", [-1, ] * len(self.dl_list))

        self.sample_val_every = trainer_config.get("sample_val_every", 100)


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

        self.register_buffer('dl_step_list', torch.Tensor([0, ] * len(self.dl_list)))
        # self.dl_step_list = [0, ] * len(self.dl_list)

        # self.valid_dl_iter_list = [cycle(self.accelerator.prepare_data_loader(valid_dl)) for valid_dl in self.valid_dl_list]
        # self.valid_dl_step_list = [0, ] * len(self.valid_dl_list)


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

        self.eval_model_every = trainer_config.get("eval_model_every", 2000)

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0: #and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            # rmtree(str(self.results_folder))
            print(f"detecting previous experiment checkpoints and results, the auto_resume: {auto_resume}")
            print(f"results folder: {self.results_folder}")

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.auto_resume = auto_resume

        if resume_path is not None:
            # self.load_model(resume_path)
            self.print(f"resuming the sheduler and the model from {resume_path}")
            # set the step according to the model's name
            self.print(f"before loading, steps: {self.steps}")
            self.load(resume_path)
            print(f"after loading, steps: {self.steps}")

            
            # restore the state of the dataloader
            new_dl_list = []
            for i, dl in enumerate(self.dl_list):
                new_dl_list.append(self.accelerator.skip_first_batches(dl, self.dl_step_list[i].item()))
            self.dl_list = new_dl_list
            self.dl_iter_list = [cycle(dl) for dl in self.dl_list]
            
        elif auto_resume:
            self.accelerator.load_state()

            # restore the state of the dataloader
            new_dl_list = []
            for i, dl in enumerate(self.dl_list):
                new_dl_list.append(self.accelerator.skip_first_batches(dl, self.dl_step_list[i].item()))
            self.dl_list = new_dl_list
            self.dl_iter_list = [cycle(dl) for dl in self.dl_list]
        else:
            print(f"nothing to resume, the auto_resume: {auto_resume}, and resume_path: {resume_path}.")
            print(f"training from scratch")
        
        self.wandb_logger = self.accelerator.get_tracker("wandb")
        

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
        # path = Path(path)
        # assert path.exists()
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
        elif batch["data_type"][0] == "imageseg":
            seg_mask = batch["seg_mask"].to(self.device)
            seg_data = batch["image"].to(self.device)
            batch["seg_mask"] = seg_mask
            batch["image"] = seg_data
        elif batch["data_type"][0] == "imageopenseg":
            seg_mask = batch["seg_mask"].to(self.device)
            seg_data = batch["image"].to(self.device)
            batch["seg_mask"] = seg_mask
            batch["image"] = seg_data
            seg_mask_promp_dict = batch["seg_mask_promp_dict"]
            for key, value in seg_mask_promp_dict.items():
                seg_mask_promp_dict[key] = value.to(self.device)
            batch["seg_mask_promp_dict"] = seg_mask_promp_dict
        else:
            raise ValueError(f"unsupported data type: {batch['data_type'][0]}, is open seg: {batch['data_type'][0] == 'imageopenseg'}")
        return batch


    def train_step_single_dataset(self, dataset_index=None, vis=False):
        batch = next(self.dl_iter_list[dataset_index])
        batch = self.prepare_batch(batch)
        #video = video
        
        with self.accelerator.accumulate(self.CTClip):
            with self.accelerator.autocast():
                if not vis:
                    loss, loss_dict = self.CTClip(batch, return_loss=True, return_loss_dict=True, 
                                                device=self.device, accelerator=self.accelerator)
                else:
                    loss, loss_dict, to_visualize = self.CTClip(batch, return_loss=True, return_loss_dict=True, 
                                                return_visualize=True, device=self.device, accelerator=self.accelerator)
                    self.wandb_logger.log(to_visualize, step=self.steps.int().item())
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
        vis_list = [False, ] * len(self.dl_list)
        for i in range(len(self.vis_train_interval)):
            if self.vis_train_interval[i] > 0 and self.dl_step_list[i].item() % self.vis_train_interval[i] == 0:
                vis_list[i] = True
        loss_dict = {}
        for i, acc_step in enumerate(acc_steps_list):
            for j in range(acc_step):
                loss_dict_single = self.train_step_single_dataset(dataset_index=i, vis = vis_list[i] and j==0) # only vis for the first step in acc_step
                self.dl_step_list[i] += 1
                loss_dict = self.loss_update(loss_dict, loss_dict_single)
        return loss_dict
    
    def eval_tests(self, models_to_evaluate):
        for model, steps, model_name in models_to_evaluate:
            for test_func in self.valid_tests:
                results = test_func(model)
                to_log_dict = results.get("log_dict", None)
                if to_log_dict is not None:
                    wandb_log_dict = {f"{model_name}_" + key: value for key, value in to_log_dict.items()}
                    print(f"wandb log dict: {wandb_log_dict}")
                    self.wandb_logger.log(wandb_log_dict, step=steps)
                to_visualize = results.get("to_visualize_dict", None)
                # log the image
                # todo: debug the image logging process
                if to_visualize is not None:
                    wandb_image_log_dict = {}
                    for key, value in to_visualize.items():
                        image_value = wandb.Image(value, caption=key)
                        wandb_image_log_dict[f"{model_name}_" + key] = image_value
                    self.wandb_logger.log(wandb_image_log_dict, step=steps)
    
    def sample_tests(self, models_to_evaluate):
        for model, steps, model_name in models_to_evaluate:
            for test_func in self.vis_val_tests:
                results = test_func(model)
                to_log_dict = results.get("log_dict", None)
                if to_log_dict is not None:
                    wandb_log_dict = {f"{model_name}_" + key: value for key, value in to_log_dict.items()}
                    print(f"wandb log dict: {wandb_log_dict}")
                    self.wandb_logger.log(wandb_log_dict, step=steps)
                to_visualize = results.get("to_vis_dict", None)
                # log the image
                # todo: debug the image logging process
                if to_visualize is not None:
                    wandb_image_log_dict = {}
                    for key, value in to_visualize.items():
                        if isinstance(value, wandb.Image):
                            img_value = value
                        elif isinstance(value, np.ndarray):
                            img_value = wandb.Image(value, cation=key)
                        else:
                            raise ValueError(f"unsupported type of value for visualization: {type(value)}")
                        wandb_image_log_dict[f"{model_name}_" + key] = img_value
                    self.wandb_logger.log(wandb_image_log_dict, step=steps)


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
        # exit()

        self.wandb_logger.log(logs, step=self.steps.int().item())

        if not ((steps + 1) % self.sample_val_every):
            if self.is_main:
                with torch.no_grad():
                    models_to_sample = ((self.CTClip, int(steps), "ctclip"),)
                    print(f"sampling eval data on model: {steps}")
                    self.sample_tests(models_to_sample)


        if not ((steps + 1) % self.eval_model_every):
            if self.is_main: 
                with torch.no_grad():
                    models_to_evaluate = ((self.CTClip, int(steps), "ctclip"),)
                    print(f"evaluating model: {steps}")
                    self.eval_tests(models_to_evaluate)


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


        if not ((steps + 1) % self.save_model_every):
            # state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
            # the following code will also work, and only get state_dict on rank0
            # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # with FSDP.state_dict_type(self.CTCLIP, StateDictType.FULL_STATE_DICT, save_policy):
            #     state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
            # if self.is_main:
            print(f"save model at step: {steps}, output_dir: {self.results_folder}")
            self.accelerator.save_state()
            print(f"successfully saved model at step: {steps}")
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
        self.accelerator.wait_for_everyone() # todo: check if this is necessary
        self.accelerator.end_training()
