import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm

from transformers import BertTokenizer

from multiprocessing import Pool

from data_inference import *

import time



def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def npz_to_tensor(path):
    # start_time = time.time()
    img_data = np.load(path)['arr_0']
    # time_1 = time.time()
    # print(f"loading npz time: {time_1 - start_time}")
    img_data= np.transpose(img_data, (1, 2, 0))
    # img_data = img_data*1000
    # hu_min, hu_max = -1000, 1000
    # img_data = np.clip(img_data, hu_min, hu_max)

    # img_data = (img_data / 1000).astype(np.float32)
    min_value, max_value = -1, 1
    img_data = np.clip(img_data, min_value, max_value)
    img_data = (img_data - min_value) / (max_value - min_value)
    img_data = img_data.astype(np.float32)

    # slices=[]

    tensor = torch.tensor(img_data)

    # time_2 = time.time()
    # print(f"converting npz to tensor time: {time_2 - time_1}")

    # Get the dimensions of the input tensor
    target_shape = (480,480,240)
    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    # time_3 = time.time()
    # print(f"padding time: {time_3 - time_2}")

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    # time_4 = time.time()
    # print(f"final processing time: {time_4 - time_3}")

    return tensor


# def npz_mask_to_tensor(path):
#     time_0 = time.time()
#     img_data = np.load(path)['arr_0']

#     time_1 = time.time()
#     print(f"loading npz mask time: {time_1 - time_0}")

#     img_data = torch.tensor(img_data.astype(np.float32))

#     img_data = torch.permute(img_data, (0, 2, 3, 1))

#     # img_data= np.transpose(img_data, (0, 2, 3, 1))
#     # img_data = img_data*1000
#     # hu_min, hu_max = -1000, 1000
#     # img_data = np.clip(img_data, hu_min, hu_max)

#     # img_data = (img_data / 1000).astype(np.float32)
#     min_value, max_value = 0, 1
#     img_data = torch.clip(img_data, min_value, max_value)
#     tensor = (img_data - min_value) / (max_value - min_value)

#     # slices=[]
#     time_2 = time.time()
#     print(f"transposing npz mask time: {time_2 - time_1}")

#     # tensor = torch.tensor(img_data)
#     # Get the dimensions of the input tensor
#     target_shape = (480,480,240)
#     # Extract dimensions
#     n_c, h, w, d = tensor.shape

#     # Calculate cropping/padding values for height, width, and depth
#     dh, dw, dd = target_shape

#     h_start = max((h - dh) // 2, 0)
#     h_end = min(h_start + dh, h)
#     w_start = max((w - dw) // 2, 0)
#     w_end = min(w_start + dw, w)
#     d_start = max((d - dd) // 2, 0)
#     d_end = min(d_start + dd, d)

#     # Crop or pad the tensor
#     tensor = tensor[:, h_start:h_end, w_start:w_end, d_start:d_end]

#     pad_h_before = (dh - tensor.size(1)) // 2
#     pad_h_after = dh - tensor.size(1) - pad_h_before

#     pad_w_before = (dw - tensor.size(2)) // 2
#     pad_w_after = dw - tensor.size(2) - pad_w_before

#     pad_d_before = (dd - tensor.size(3)) // 2
#     pad_d_after = dd - tensor.size(3) - pad_d_before

#     # print(f"before padding: {tensor.shape}")

#     tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=0)

#     # print(f"after padding: {tensor.shape}")

#     tensor = tensor.permute(0, 3, 1, 2)

#     tensor = tensor #.unsqueeze(0)

#     time_3 = time.time()
#     print(f"final processing mask time: {time_3 - time_2}")

#     return tensor

def npz_mask_to_tensor(path):

    start_time = time.time()
    img_data = np.load(path)['arr_0']

    # time_1 = time.time()
    # print(f"loading npz mask time: {time_1 - start_time}")

    img_data = torch.tensor(img_data.astype(np.float32))

    # img_data = torch.permute(img_data, (0, 2, 3, 1))
    min_value, max_value = 0, 1
    img_data = torch.clip(img_data, min_value, max_value)
    tensor = (img_data - min_value) / (max_value - min_value)
    target_shape = (240, 480, 480)
    # Extract dimensions
    n_c, h, w, d = tensor.shape
    
    dh, dw, dd = target_shape

    # print(f"padding mask tensor: {tensor.shape}")
    # assert h == dh and w == dw and d == dd, f"mask shape not equal to target shape: {h, w, d} vs {dh, dw, dd}"

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[:, h_start:h_end, w_start:w_end, d_start:d_end]

    # time_2 = time.time()
    # print(f"cropping mask time: {time_2 - time_1}")

    pad_h_before = (dh - tensor.size(1)) // 2
    pad_h_after = dh - tensor.size(1) - pad_h_before

    pad_w_before = (dw - tensor.size(2)) // 2
    pad_w_after = dw - tensor.size(2) - pad_w_before

    pad_d_before = (dd - tensor.size(3)) // 2
    pad_d_after = dd - tensor.size(3) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=0)

    # time_3 = time.time()
    # print(f"padding mask time: {time_3 - time_2}")

    tensor = tensor

    return tensor

class CTReportDataset(Dataset):
    """
    now use the preprocessed train npz files to load the data, no need to resize the nii files
    """
    def __init__(self, data_folder, csv_file, metadata_train=None):
        self.data_folder = data_folder
        self.cache_data_list_folder = os.path.join(data_folder, './tmp_cache_data_list')
        os.makedirs(self.cache_data_list_folder, exist_ok=True)
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        percent = 80
        num_files = int((len(self.samples) * percent) / 100)
        self.samples = self.samples[:num_files]
        print(len(self.samples))
        self.count = 0

        self.metadata = pd.read_csv(metadata_train)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_number = row['VolumeName'].split("/")[-1]
            accession_to_text[accession_number] = row["Findings_EN"],row['Impressions_EN']

        return accession_to_text


    # since the data is stored in mnt and the loading is too slow for training dataset, so the multi-preocessing is used

    def process_patient_folder(self, patient_folder, accession_to_text, paths):
        samples = []
        accession_folders = glob.glob(os.path.join(patient_folder, '*'))
        for accession_folder in accession_folders:
            nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))
            # print(f"nii files: {nii_files}, accession folder:{accession_folder}")
            for nii_file in nii_files:
                accession_number = nii_file.split("/")[-1].replace(".npz", ".nii.gz")
                if accession_number not in accession_to_text:
                    continue

                impression_text = accession_to_text[accession_number]
                if impression_text == "Not given.":
                    impression_text = ""

                input_text_concat = "".join(str(text) for text in impression_text) if impression_text else ""
                samples.append((nii_file, input_text_concat))
                # print(f"appending sample:{nii_file}")
                paths.append(nii_file)
        return samples

    def get_file_by_walk(self, root_folder, ext=".npz"):
        npz_file_lists = []
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(ext):
                    npz_file_lists.append(os.path.join(root, file))
        return npz_file_lists

    def process_patient_npz(self, npz_file_lists, accession_to_text, paths):
        samples = []
        for npz_file in npz_file_lists:
            accession_number = npz_file.split("/")[-1].replace(".npz", ".nii.gz")
            if accession_number not in accession_to_text:
                continue

            impression_text = accession_to_text[accession_number]
            if impression_text == "Not given.":
                impression_text = ""

            input_text_concat = "".join(str(text) for text in impression_text) if impression_text else ""
            samples.append((npz_file, input_text_concat))
            paths.append(npz_file)
        return samples


    def prepare_samples(self):
        if os.path.exists(os.path.join(self.cache_data_list_folder, 'image_samples.txt')) and os.path.exists(os.path.join(self.cache_data_list_folder, 'report_samples.txt')):
            with open(os.path.join(self.cache_data_list_folder, 'image_samples.txt'), 'r') as f:
                image_samples_name = f.readlines()
            image_samples = [sample.strip() for sample in image_samples_name]
            with open(os.path.join(self.cache_data_list_folder, 'report_samples.txt'), 'r') as f:
                report_samples_name = f.readlines()
            report_samples = [sample.strip() for sample in report_samples_name]
            samples = list(zip(image_samples, report_samples))
            print(f"finished preparing samples with cache txt within the ct report dataset, the number of samples: {len(samples)}")
            return samples
        else:
            # patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
            # print(f"start prepraring samples")
            # # with Pool() as pool:
            # #     # Use a lambda or partial to pass additional arguments
            # #     results = pool.starmap(self.process_patient _folder, [(folder, self.accession_to_text, self.paths) for folder in patient_folders])

            # results = []
            # for patient_folder in patient_folders:
            #     samples = self.process_patient_folder(patient_folder, self.accession_to_text, self.paths)
            #     results.append(samples)

            print(f"start prepraring samples")

            npz_file_lists = self.get_file_by_walk(self.data_folder, ext=".npz")
            results = self.process_patient_npz(npz_file_lists, self.accession_to_text, self.paths)

            # Combine all patient folder results
            samples = results
            print(f"finished preparing samples, the number of samples: {len(samples)}")
            # Save the samples to cache
            image_samples = [sample[0] for sample in samples]
            report_samples = [sample[1] for sample in samples]
            with open(os.path.join(self.cache_data_list_folder, 'image_samples.txt'), 'w') as f:
                for sample in image_samples:
                    f.write(f"{sample}\n")
            with open(os.path.join(self.cache_data_list_folder, 'report_samples.txt'), 'w') as f:
                for sample in report_samples:
                    f.write(f"{sample}\n")
            return samples

    def __len__(self):
        return len(self.samples)


    # here we load the pre-processed data samples to avoid interpolation
    

    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        video_tensor = npz_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return {"image": video_tensor, "text": input_text, "data_type": "imagereport"}


class CTSegDataset(Dataset):
    """
    the ct seg dataset, used for both training and validation, as the image and seg mask data are returned
    """
    def __init__(self, data_folder, mask_folder):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.cache_data_list_folder = os.path.join(data_folder, './tmp_cache_data_list')
        self.cache_mask_list_folder = os.path.join(mask_folder, './tmp_cache_mask_list')
        os.makedirs(self.cache_data_list_folder, exist_ok=True)
        os.makedirs(self.cache_mask_list_folder, exist_ok=True)
        # self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        # percent = 100
        # num_files = int((len(self.samples) * percent) / 100)
        # self.samples = self.samples[:num_files]
        # print(len(self.samples))
        # self.count = 0
    
    def __len__(self):
        return len(self.samples)
    
    # data lie in data_folder/*.npz, mask lie in mask_folder/*.npz
    def prepare_samples(self):
        img_sample_txt_path = os.path.join(self.cache_data_list_folder, 'image_samples.txt')
        mask_sample_txt_path = os.path.join(self.cache_mask_list_folder, 'mask_samples.txt')
        if os.path.exists(img_sample_txt_path) and os.path.exists(mask_sample_txt_path):
            with open(img_sample_txt_path, 'r') as f:
                img_samples_name = f.readlines()
            img_samples = [sample.strip() for sample in img_samples_name]
            with open(mask_sample_txt_path, 'r') as f:
                mask_samples_name = f.readlines()
            mask_samples = [sample.strip() for sample in mask_samples_name]
            samples = list(zip(img_samples, mask_samples))
            print(f"finished preparing samples with cache txt within the ct seg dataset, the number of samples: {len(samples)}")
            return samples
        else:
            data_names = glob.glob(os.path.join(self.data_folder, '*.npz'))
            mask_names = glob.glob(os.path.join(self.mask_folder, '*.npz'))
            assert len(data_names) == len(mask_names)
            samples = list(zip(data_names, mask_names))
            with open(img_sample_txt_path, 'w') as f:
                for sample in data_names:
                    f.write(f"{sample}\n")
            with open(mask_sample_txt_path, 'w') as f:
                for sample in mask_names:
                    f.write(f"{sample}\n")
        return samples

    def __getitem__(self, index):
        
        data_file, mask_file = self.samples[index]
        # the seg data is already preprocessed, no need to resize, pad, just load
        try:
            video_tensor = torch.tensor(np.load(data_file)['arr_0']).unsqueeze(0) # missing channel dim in the saved data
            mask_tensor = torch.tensor(np.load(mask_file)['arr_0'])
        except Exception as e:
            print(f"error loading seg data: {e} for data file: {data_file}")
            print(f"mask file: {mask_file}")

        # return video_tensor, mask_tensor
        return {"image": video_tensor, "seg_mask": mask_tensor, 
                "data_type": "imageseg"}


class CTOpenSegDataset(Dataset):
    """
    the ct seg dataset, used for both training and validation, as the image and seg mask data are returned
    """
    def __init__(self, data_folder, mask_folder, seg_mask_name_table, seg_mask_prompt_type="this_region"):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.seg_mask_name_dict = self.read_xlsx(seg_mask_name_table)
        self.tokenizer= BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.seg_prompt_dict = self.load_seg_prompt_dict(seg_mask_prompt_type, self.seg_mask_name_dict)
        self.cache_data_list_folder = os.path.join(data_folder, './tmp_cache_data_list')
        self.cache_mask_list_folder = os.path.join(mask_folder, './tmp_cache_mask_list')
        os.makedirs(self.cache_data_list_folder, exist_ok=True)
        os.makedirs(self.cache_mask_list_folder, exist_ok=True)
        # self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        # percent = 100
        # num_files = int((len(self.samples) * percent) / 100)
        # self.samples = self.samples[:num_files]
        # print(len(self.samples))
        # self.count = 0

    def read_xlsx(self, seg_mask_name_table):
        df = pd.read_excel(seg_mask_name_table)
        df_ids = df["ID"].tolist()
        # all to int
        df_ids = [int(i) for i in df_ids]
        df_name = df["NAME"].tolist()
        # create dict
        df_dict = dict(zip(df_ids, df_name))
        print(f"loaded seg mask name table: {df_dict}")
        return df_dict

    def load_seg_prompt_dict(self, seg_mask_prompt_type, seg_mask_name_dict):
        if seg_mask_prompt_type == "this_region":
            seg_mask_prompt_dict = {}
            for key, value in seg_mask_name_dict.items():
                prompt = f"This is region of {value}."
                tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                seg_mask_prompt_dict[key] = tokens.input_ids
            return seg_mask_prompt_dict
        elif seg_mask_prompt_type == "this_is":
            seg_mask_prompt_dict = {}
            for key, value in seg_mask_name_dict.items():
                prompt = f"This is {value}."
                tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                seg_mask_prompt_dict[key] = tokens.input_ids.squeeze(0)
                # print(f"input ids shape: {tokens.input_ids.shape}")
            return seg_mask_prompt_dict
        else:
            raise ValueError(f"Unknown seg mask prompt type: {seg_mask_prompt_type}")
    
    def __len__(self):
        return len(self.samples)
    
    # data lie in data_folder/*.npz, mask lie in mask_folder/*.npz
    def prepare_samples(self):
        img_sample_txt_path = os.path.join(self.cache_data_list_folder, 'image_samples.txt')
        mask_sample_txt_path = os.path.join(self.cache_mask_list_folder, 'mask_samples.txt')
        if os.path.exists(img_sample_txt_path) and os.path.exists(mask_sample_txt_path):
            with open(img_sample_txt_path, 'r') as f:
                img_samples_name = f.readlines()
            img_samples = [sample.strip() for sample in img_samples_name]
            with open(mask_sample_txt_path, 'r') as f:
                mask_samples_name = f.readlines()
            mask_samples = [sample.strip() for sample in mask_samples_name]
            samples = list(zip(img_samples, mask_samples))
            print(f"finished preparing samples with cache txt within the open seg dataset, the number of samples: {len(samples)}")
            return samples
        else:
            print(f"start prepraring samples for open seg dataset")
            data_names = glob.glob(os.path.join(self.data_folder, '*.npz'))
            mask_names = glob.glob(os.path.join(self.mask_folder, '*.npz'))
            assert len(data_names) == len(mask_names)
            samples = list(zip(data_names, mask_names))
            with open(img_sample_txt_path, 'w') as f:
                for sample in data_names:
                    f.write(f"{sample}\n")
            with open(mask_sample_txt_path, 'w') as f:
                for sample in mask_names:
                    f.write(f"{sample}\n")
            print(f"finished preparing samples for open seg dataset, the number of samples: {len(samples)}")
        return samples

    def __getitem__(self, index):
        start_time = time.time()
        data_file, mask_file = self.samples[index]
        # the seg data is already preprocessed, no need to resize, pad, just load
        # try:
        # video_tensor = torch.tensor(np.load(data_file)['arr_0'].unsqueeze(0) # missing channel dim in the saved data
        video_tensor = npz_to_tensor(data_file)
        # step1 = time.time()
        # print(f"step1 in loading: {step1 - start_time}")
        # mask_tensor = torch.tensor(np.load(mask_file)['arr_0'])
        mask_tensor = npz_mask_to_tensor(mask_file)
        # step2 = time.time()
        # print(f"step2 in loading: {step2 - step1}")
        # except Exception as e:
        #     print(f"error loading seg data: {e} for data file: {data_file}")
        #     print(f"mask file: {mask_file}")

        # return video_tensor, mask_tensor
        return {"image": video_tensor, "seg_mask": mask_tensor, 
                "data_type": "imageopenseg", "seg_mask_promp_dict": self.seg_prompt_dict}



class InfiniteCycleSampler(Sampler):
    """
    A sampler that infinitely loops through the dataset.
    """
    def __init__(self, dataset):
        """
        Args:
            dataset (torch.utils.data.Dataset): Dataset to sample from.
        """
        self.dataset = dataset
        self.dataset_length = len(dataset)

    def __iter__(self):
        """
        Infinite generator that loops through dataset indices.
        """
        while True:
            # Shuffle dataset indices for randomness (optional)
            indices = torch.randperm(self.dataset_length).tolist()
            yield from indices

    def __len__(self):
        """
        Returns the dataset length, though it's infinite by design.
        """
        return float('inf')

def create_train_ds(config):
    if config["type"] == "imagereport":
        return CTReportDataset(config["data_train"], config["reports_file_train"], config["metadata_train"])
    elif config["type"] == "imageseg":
        return CTSegDataset(config["seg_data_train"], config["seg_mask_train"])
    elif config["type"] == "imageopenseg":
        return CTOpenSegDataset(config["seg_data_train"], config["seg_mask_train"], config["seg_mask_name_table"], config["seg_mask_prompt_type"])
    else:
        raise ValueError(f"Unknown dataset type: {config['type']}")


def create_train_dl(train_ds, train_dl_config):
    # if train_dl_config["sampler_type"] == "InfinteCycleSampler":
    #     sampler = InfiniteCycleSampler(train_ds)
    # else:
    #     sampler = None
    return DataLoader(train_ds, batch_size=train_dl_config["batch_size"], num_workers=train_dl_config["num_workers"], shuffle=True)


def create_valid_ds(config):
    if config["type"] == "imagereport":
        return CTReportDatasetinfer(config["data_valid"], config["reports_file_valid"], labels = config["labels"])
    elif config["type"] == "imageopenseg":
        return CTOpenSegDataset(config["seg_data_valid"], config["seg_mask_valid"], config["seg_mask_name_table"], config["seg_mask_prompt_type"])
    else:
        raise ValueError(f"Unknown dataset type: {config['type']}")


def create_valid_dl(valid_ds, valid_dl_config):
    return DataLoader(valid_ds, batch_size=valid_dl_config["batch_size"], shuffle=False, num_workers=valid_dl_config["num_workers"])


def create_train_dl_list(train_dl_config):
    ds_list = [create_train_ds(train_dl_config) for train_dl_config in train_dl_config]
    # print(f"ds list: {ds_list}")
    dl_list = [create_train_dl(ds, train_dl_config) for ds, train_dl_config in zip(ds_list, train_dl_config)]
    return dl_list
    

def create_valid_dl_list(valid_dl_config):
    ds_list = [create_valid_ds(valid_ds_config) for valid_ds_config in valid_dl_config]
    dl_list = [create_valid_dl(ds, valid_dl_config) for ds, valid_dl_config in zip(ds_list, valid_dl_config)]
    return dl_list  # Return the created list



# class CombinedDataset(Dataset):
#     def __init__(self, data_folder=None, csv_file=None, metadata_train=None,
#                  seg_data=None, seg_mask=None):
#         self.report_dataset = CTReportDataset(data_folder, csv_file, metadata_train)
#         self.seg_dataset = CTSegDataset(seg_data, seg_mask)
#         self.n_image_txt_pairs = len(self.report_dataset)
#         self.n_image_seg_pairs = len(self.seg_dataset)
#         self.dummy_seg = self.find_dummy_seg()
#         self.dummy_text = self.find_dummy_text()

#     def find_dummy_seg(self):
#         # load the 1st seg data to find the dummy seg
#         dummy_seg = self.seg_dataset[0]["seg_mask"] #* 0
#         return dummy_seg

#     def find_dummy_text(self):
#         # load the 1st text data to find the dummy text
#         dummy_text = self.report_dataset[0]["text"] #* 0
#         return dummy_text

    
#     def __len__(self):
#         num_report = sum([len(report) for report in self.report_dataset])
#         num_seg = sum([len(seg) for seg in self.seg_dataset])
#         return num_report + num_seg

#     def __getitem__(self, index):
#         # report first, then seg
#         if index < len(self.report_dataset):
#             ori_report_dict = self.report_dataset[index]
#             # add dummy seg
#             ori_report_dict["seg_mask"] = self.dummy_seg
#             return ori_report_dict
#         else:
#             ori_seg_dict = self.seg_dataset[index - len(self.report_dataset)]
#             # add dummy text
#             ori_seg_dict["text"] = self.dummy_text
#             return ori_seg_dict
