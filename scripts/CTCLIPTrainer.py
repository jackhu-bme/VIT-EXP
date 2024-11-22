from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data import CTReportDataset
from data_inference import CTReportDatasetinfer

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

import wandb

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



def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

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

class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        batch_size,
        data_train = "train",
        data_valid = "valid",
        reports_file_train = "data_reports.xslx",
        reports_file_valid = "data_reports.xslx",
        labels = "labels.csv",
        tokenizer = None,
        lr = 1.25e-6,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1000,
        save_model_every = 1000 ,
        results_folder = '/shares/menze.dqbm.uzh/ihamam/ctclip/',
        num_workers = 8,
        accelerate_kwargs: dict = dict(),
        resume_path = None,
        metadata_train = "train_metadata.csv",
        wandb_logger = None,
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            self.tokenizer=BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        # Load the pre-trained weights
        self.ds = CTReportDataset(data_folder=data_train, csv_file=reports_file_train, metadata_train=metadata_train)

        self.valid_ds = CTReportDatasetinfer(data_folder=data_valid, csv_file=reports_file_valid, labels = labels)


        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = True,
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers=num_workers,
            batch_size=1,
            shuffle = False,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
 			self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        if resume_path is not None:
            # self.load_model(resume_path)
            self.print(f"resuming the sheduler and the model from {resume_path}")
            # set the step according to the model's name
            self.print(f"before loading, steps: {self.steps}")
            self.resume_step = int(os.path.basename(resume_path).split(".")[-2])
            self.steps += self.resume_step
            self.print(f"resuming from step {self.steps} according to the model's name: {resume_path}")
            # restore the state of the dataloader
            self.dl = accelerate.skip_first_batches(self.dl, self.steps)
        else:
            self.resume_step = None
        
        self.wandb_logger = wandb_logger
        

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

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
        pkg = torch.load(path)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        # tmp gpu memroy check for debug only
        # continue_training = input("Continue training? (y/n)")
        # if continue_training == "n":
        #     raise Exception("Training stopped by user")

        # print(f"start training step {steps}")

        self.CTClip.train()

        # logs
        logs = {}

        # update CTClip model
        video, text = next(self.dl_iter)
        # print(video.shape)
        device=self.device
        video=video.to(device)
        mask = torch.ones((video.shape[0], video.shape[2])).bool().to(device)
        #text = text.to(device)
        text = list(text)
        text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

        #video = video
        with self.accelerator.accumulate(self.CTClip):
            with self.accelerator.autocast():
                loss, loss_dict = self.CTClip(text_tokens, video, return_loss=True, return_loss_dict=True, device=device)

        self.accelerator.backward(loss)
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
        # self.accelerator.log(logs, step=self.steps.int().item())

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


        # save model every so often

        # if self.is_main and not (steps % self.save_model_every):
        #     # print(f"Saving model at step {steps}")
        #     model_path = str(self.results_folder / f'CTClip.{steps}.pt')
        #     state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)

        #     self.accelerator.save(state_dict, model_path)

        #     # print(f"finished saving model at step {steps}")

        #     self.print(f'{steps}: saving model to {str(self.results_folder)}')

        if not (steps % self.save_model_every):
            state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
            # the following code will also work, and only get state_dict on rank0
            # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            # with FSDP.state_dict_type(self.CTCLIP, StateDictType.FULL_STATE_DICT, save_policy):
            #     state_dict=self.accelerator.get_state_dict(self.CTClip, unwrap=False)
            if self.is_main:
                model_path = str(self.results_folder / f'CTClip.{steps}.pt')
                self.accelerator.save(state_dict, model_path)

        self.steps += 1
        return logs



    # def train(self, log_fn=noop):
    #     device = next(self.CTClip.parameters()).device
    #     device=torch.device('cuda')
    #     while self.steps < self.num_train_steps:
    #         logs = self.train_step()
    #         log_fn(logs)

    #     self.print('training complete')

    def train(self, log_fn=noop):
        # device = next(self.CTClip.parameters()).device
        # device = torch.device('cuda')
        
        # 创建 tqdm 进度条
        with tqdm(total=self.num_train_steps, desc='Training', unit='step') as pbar:
            if self.resume_step is not None:
                pbar.update(self.resume_step)
            while self.steps < self.num_train_steps:
                logs = self.train_step()
                log_fn(logs)
                pbar.update(1)  # 更新进度条
