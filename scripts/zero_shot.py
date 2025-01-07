from pathlib import Path
from shutil import rmtree
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import torchvision

# from data_inference_nii import CTReportDatasetinfer
from data_inference import CTReportDatasetinfer
from data import CTSegDataset
#from data_external_valid import CTReportDatasetinfer
import numpy as np
import tqdm
import pandas as pd

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP

import os


import time

# helpers

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

class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        batch_size,
        data_folder: "external_valid",
        reports_file: "data_reports.xslx",
        lr = 1e-4,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 100,
        save_model_every = 2000,
        results_folder = './results',
        labels = "labels.csv",
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        # Load the pre-trained weights
        self.ds = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file,labels=labels)

        # Split dataset into train and validation sets


        self.dl = DataLoader(
            self.ds,
            num_workers=6,
            batch_size=1,
            shuffle = True,
        )
        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                  T_0=4000000,    # Maximum number of iterations
                                                  T_warmup=10000, # Number of warmup steps
                                                  eta_max=lr)   # Maximum learning rate


        (
 			self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)


    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

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
        # logs
        logs = {}
        if True:
            with torch.no_grad():

                models_to_evaluate = ((self.CTClip, str(steps)),)

                for model, filename in models_to_evaluate:
                    model.eval()
                    predictedall=[]
                    realall=[]
                    logits = []

                    text_latent_list = []
                    image_latent_list = []
                    accession_names=[]
                    pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
                    for i in tqdm.tqdm(range(len(self.ds))):
                        valid_data, text, onehotlabels, acc_name = next(self.dl_iter)

                        plotdir = self.result_folder_txt
                        Path(plotdir).mkdir(parents=True, exist_ok=True)

                        predictedlabels=[]
                        onehotlabels_append=[]

                        for pathology in pathologies:
                            text = [f"{pathology} is present.", f"{pathology} is not present."]
                            text_tokens=self.tokenizer(
                                            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                            output = model(text_tokens, valid_data.cuda(),  device=device)

                            output = apply_softmax(output)

                            append_out=output.detach().cpu().numpy()
                            predictedlabels.append(append_out[0])

                        predictedall.append(predictedlabels)
                        realall.append(onehotlabels.detach().cpu().numpy()[0])
                        accession_names.append(acc_name[0])
                        print(f"finished {i} out of {len(self.ds)}")

                    realall=np.array(realall)
                    predictedall=np.array(predictedall)

                    print(f"saving results to {plotdir}")

                    np.savez(f"{plotdir}labels_weights.npz", data=realall)
                    np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
                    with open(f"{plotdir}accessions.txt", "w") as file:
                        for item in accession_names:
                            file.write(item + "\n")


                    dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)

                    writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

                    dfs.to_excel(writer, sheet_name='Sheet1', index=False)

                    writer.close()
        self.steps += 1
        return logs




    def infer(self, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('Inference complete')



def ctclip_image_report_zero_shot_cls_test(ctclip):
    """
    ctclip: CTCLIP model
    """
    ctclip = ctclip.to(torch.device('cuda:0'))
    data_folder = '/mnt/input/CT-RATE/organized_dataset/val_images_preprocessed'
    reports_file= "/mnt/input/CT-RATE/organized_dataset/csv_dir/reports/validation_reports.csv"
    labels = "/mnt/input/CT-RATE/organized_dataset/csv_dir/labels/valid_predicted_labels.csv"
    batch_size = 1
    num_train_steps = 1
    inference = CTClipInferenceFast(
        ctclip,
        data_folder = data_folder,
        reports_file= reports_file,
        labels = labels,
        batch_size = batch_size,
        results_folder="./",
        num_train_steps = num_train_steps,
    )
    results_dict = inference.infer_return_res_dict() # the dict of results to be logged directly with wandb (todo: remember to log the dict and the training step!)
    return results_dict



class CTClipInferenceFast(nn.Module):
    """
    A faster version for inference on single gpu
    
    The reasons for being faster are:
    1) for text embeddings, we embed the 18 pathologies for only once during the whole inference process as it is pre-trained
    2) for image embeddings, we embed the images for only once for each image
    3) when inference, change the dataloader to multi-threaded dataloader with a bigger batch size for faster inference

    finished versions:
    1): done
    2): done
    3): not done
    
    """
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        batch_size,
        data_folder: "external_valid",
        reports_file: "data_reports.xslx",
        lr = 1e-4,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 100,
        save_model_every = 2000,
        results_folder = './results',
        labels = "labels.csv",
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        # Load the pre-trained weights
        self.ds = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file,labels=labels)

        # Split dataset into train and validation sets


        self.dl = DataLoader(
            self.ds,
            num_workers=16,
            batch_size=1,
            shuffle = True,
        )
        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                  T_0=4000000,    # Maximum number of iterations
                                                  T_warmup=10000, # Number of warmup steps
                                                  eta_max=lr)   # Maximum learning rate


        (
 			self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.prepare_infer()

    def prepare_infer(self):
        patho_txtt_list = []
        pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 
                       'Pericardial effusion','Coronary artery wall calcification', 
                       'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 
                       'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 
                       'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 
                       'Bronchiectasis','Interlobular septal thickening']
        for pathology in pathologies:
            text = [f"{pathology} is present.", f"{pathology} is not present."]
            text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)
            # self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
            text_embed = self.CTClip.text_transformer(text_tokens.input_ids, text_tokens.attention_mask)
            patho_txtt_list.append({"pathology": pathology, "text_tokens": text_tokens, "text_embed": text_embed})
        self.patho_txtt_list = patho_txtt_list

        self.patho_list = pathologies



    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

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

    def train_step(self, save_results=True, return_dict=False):
        steps = int(self.steps.item())
        # logs
        logs = {}
        with torch.no_grad():
            # models_to_evaluate = ((self.CTClip, str(steps)),)
            # for model, filename in models_to_evaluate:
            model = self.CTClip
            model.eval()
            predictedall=[]
            realalltmp=[]
            accession_names=[]
            
            for i in tqdm.tqdm(range(len(self.ds))):
                if i > 10:
                    break
                valid_data, text, onehotlabels, acc_name = next(self.dl_iter)

                valid_data = valid_data.cuda()

                # enc_image= self.visual_transformer(image, return_encoded_tokens=True)
                image_embed = model.visual_transformer(valid_data, return_encoded_tokens=True)
                plotdir = self.result_folder_txt
                Path(plotdir).mkdir(parents=True, exist_ok=True)

                predictedlabels=[]
                onehotlabels_append=[]

                for i, patho_txtt in enumerate(self.patho_txtt_list):
                    # patho_txtt = self.patho_txtt_list[i]
                    # pathology = patho_txtt["pathology"]
                    text_tokens = patho_txtt["text_tokens"]
                    text_embed = patho_txtt["text_embed"]

                    output = model.forward_infer(text_tokens, valid_data, buffer_text_embed=text_embed, buffer_image_embed=image_embed)
                    output = apply_softmax(output)             
                    # print(f"output: {output}")
                    # append_out=output.detach().cpu().numpy()
                    # print("a out 0: ", append_out[0])
                    predictedlabels.append(output[0])
                    # step_3_time = time.time() - step_2_time - start_time
                    # print(f"step 3 time: {step_3_time}")             
                predictedall.append(predictedlabels)
                # print(f"one hot labels in the loop: {onehotlabels}")
                realalltmp.append(onehotlabels[0])
                accession_names.append(acc_name[0])

                # exit()
            
            realall = []
            for labels in realalltmp:
                labels = labels.detach().cpu().numpy()
                realall.append(labels)
            realall=np.array(realall)
            # final load the labels from gpu to cpu
            for labels in predictedall:
                for i in range(len(labels)):
                    labels[i] = labels[i].detach().cpu().numpy()
            print(f"predictedall: {predictedall}")
            predictedall=np.array(predictedall)

            if save_results:
                print(f"saving results to {plotdir}")
                np.savez(f"{plotdir}labels_weights.npz", data=realall)
                np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
                with open(f"{plotdir}accessions.txt", "w") as file:
                    for item in accession_names:
                        file.write(item + "\n")

            dfs = evaluate_internal(predictedall,realall, self.patho_list, plotdir)
            if save_results:
                writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')
                dfs.to_excel(writer, sheet_name='Sheet1', index=False)
                writer.close()
        self.steps += 1
        if return_dict:
            roc_dict = {
                col_name: dfs[col_name].iloc[0]
                for col_name in dfs.columns
            }
            return {"log_dict": roc_dict}
        else:
            return logs



    def infer(self, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        logs = self.train_step()
        log_fn(logs)
        self.print('Inference complete')

    def infer_return_res_dict(self):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        result_dict = self.train_step(return_dict=True, save_results=False)
        return result_dict


class CTClipInferenceSeg(nn.Module):
    """
    inference on seg dataset, to calculate the segmentation results and output visualization maps
    
    """
    def __init__(
        self,
        CTClip: CTCLIP,
        config=None,
        # *,
        # num_train_steps,
        # batch_size,
        # data_folder: "external_valid",
        # reports_file: "data_reports.xslx",
        # lr = 1e-4,
        # wd = 0.,
        # max_grad_norm = 0.5,
        # save_results_every = 100,
        # save_model_every = 2000,
        results_folder = './results',
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        # self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        data_config = config["valid_data_list"][1] # todo: ref for a more general way to get the seg data

        # self.num_train_steps = num_train_steps
        # self.batch_size = batch_size

        all_parameters = set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=0., wd=0.)

        # self.max_grad_norm = max_grad_norm
        # self.lr=lr
        # Load the pre-trained weights
        # self.ds = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file,labels=labels)
        seg_data_valid_folder = data_config["seg_data_valid"]
        seg_mask_valid_folder = data_config["seg_mask_valid"]
        print(f"seg_data_valid_folder: {seg_data_valid_folder}, seg_mask_valid_folder: {seg_mask_valid_folder}")
        self.ds = CTSegDataset(data_folder=seg_data_valid_folder, mask_folder=seg_mask_valid_folder)


        # Split dataset into train and validation sets
        self.dl = DataLoader(
            self.ds,
            num_workers=0,
            batch_size=1,
            shuffle = False,
        )
        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        # self.device = self.accelerator.device
        # self.CTClip.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optim,
                                                  T_0=4000000,    # Maximum number of iterations
                                                  T_warmup=10000, # Number of warmup steps
                                                  eta_max=0.)   # Maximum learning rate


        (
 			self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
            self.optim,
            self.lr_scheduler
        )

        # self.save_model_every = save_model_every
        # self.save_results_every = save_results_every
        self.result_folder_txt = self.results_folder
        self.result_folder_vis = os.path.join(self.results_folder, "vis_maps")
        os.makedirs(self.result_folder_vis, exist_ok=True)
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)





    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

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

    
    def visualize(self, vis_slices, vis_name):
        # a visualize slice is of shape (C, H, W, 3, 3)
        # create a meshgrid of the slices, the meshgrid is of shape (C, 3*3), the output image shape is (C*H, 3*3*W)
        # use the torchvision.utils.make_grid to make the slices into a grid
        
        # vis_slices = vis_slices.permute((0, 3, 4, 1, 2))

        # vis_shape = vis_slices.shape

        # grid_slices = torchvision.utils.make_grid(vis_slices, nrow=vis_shape[0], padding=0)
        # # save
        # torchvision.utils.save_image(grid_slices, vis_name)
        pass # no vis due to time limits


        

    def train_step(self):
        # logs
        logs = {}
        with torch.no_grad():
            # for model, filename in models_to_evaluate:
            self.CTClip.eval()
            dice_scores = []    # save for each channel, corresponding to each seg label
            
            for i in tqdm.tqdm(range(len(self.ds))):
                # if i > 10:
                #     break # for debug only
                # try:
                batch = next(self.dl_iter)

                batch["image"] = batch["image"].cuda()
                batch["seg_mask"] = batch["seg_mask"].cuda()
                # print(f"image name: {batch['img_name']}")
                seg_loss, loss_dict, metrics_dict, vis_dict = self.CTClip(batch, return_metrics=True, return_vis=True)
                print(f"for batch {i}, dice score is {metrics_dict['dice_score']}")
                plotdir = self.result_folder_txt
                Path(plotdir).mkdir(parents=True, exist_ok=True)
                dice_scores.append(metrics_dict["dice_score"])
                axial_vis_slices = vis_dict["axial_slices"]
                coronal_vis_slices = vis_dict["coronal_slices"]
                sagittal_vis_slices = vis_dict["sagittal_slices"]
                # plot the vis maps
                axial_name = f"axial_image_{i}.png"
                coronal_name = f"coronal_image_{i}.png"
                sagittal_name = f"sagittal_image_{i}.png"
                axial_path = os.path.join(self.result_folder_vis, axial_name)
                coronal_path = os.path.join(self.result_folder_vis, coronal_name)
                sagittal_path = os.path.join(self.result_folder_vis, sagittal_name)
                self.visualize(axial_vis_slices, axial_path)
                self.visualize(coronal_vis_slices, coronal_path)
                self.visualize(sagittal_vis_slices, sagittal_path)
                # except Exception as e:
                #     print(f"error in batch {i}: {e}")
                #     continue
            total_dice_scores = np.stack(dice_scores, axis=0)
            # compute mean for each class
            mean_dice_scores = np.mean(total_dice_scores, axis=0)
            print(f"mean dice scores: {mean_dice_scores}")
            # save as npy and write to txt
            np.save(os.path.join(plotdir, "dice_scores.npy"), mean_dice_scores)
            np.savetxt(os.path.join(plotdir, "dice_scores.txt"), mean_dice_scores)
               
        return logs




    def infer(self, log_fn=noop):
        logs = self.train_step()
        log_fn(logs)

        self.print('Inference complete')





