import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm

from multiprocessing import Pool

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

class CTReportDataset(Dataset):
    """
    now use the preprocessed train npz files to load the data, no need to resize the nii files
    """
    def __init__(self, data_folder, csv_file, metadata_train=None, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.cache_data_list_folder = os.path.join(data_folder, './cache_data_list')
        os.makedirs(self.cache_data_list_folder, exist_ok=True)
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        percent = 80
        num_files = int((len(self.samples) * percent) / 100)
        #num_files = 2286
        self.samples = self.samples[:num_files]
        print(len(self.samples))
        self.count = 0

        # "/mnt/input/CT-RATE/organized_dataset/csv_dir/metadata/train_metadata.csv"
        self.metadata = pd.read_csv(metadata_train)

        #self.resize_dim = resize_dim
        #self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_number = row['VolumeName'].split("/")[-1]
            accession_to_text[accession_number] = row["Findings_EN"],row['Impressions_EN']

        return accession_to_text


    # def prepare_samples(self):
    #     samples = []
    #     for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
    #         for accession_folder in glob.glob(os.path.join(patient_folder, '*')):

    #             for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
    #                 accession_number = nii_file.split("/")[-1]
    #                 #accession_number = accession_number.replace(".npz", ".nii.gz")
    #                 if accession_number not in self.accession_to_text:
    #                     continue

    #                 impression_text = self.accession_to_text[accession_number]

    #                 if impression_text == "Not given.":
    #                     impression_text=""

    #                 input_text_concat = ""
    #                 for text in impression_text:
    #                     input_text_concat = input_text_concat + str(text)
    #                 input_text_concat = impression_text[0]
    #                 input_text = f'{impression_text}'
    #                 samples.append((nii_file, input_text_concat))
    #                 self.paths.append(nii_file)
    #     return samples

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

    def prepare_samples(self):
        if os.path.exists(os.path.join(self.cache_data_list_folder, 'image_samples.txt')) and os.path.exists(os.path.join(self.cache_data_list_folder, 'report_samples.txt')):
            with open(os.path.join(self.cache_data_list_folder, 'image_samples.txt'), 'r') as f:
                image_samples_name = f.readlines()
            image_samples = [sample.strip() for sample in image_samples_name]
            with open(os.path.join(self.cache_data_list_folder, 'report_samples.txt'), 'r') as f:
                report_samples_name = f.readlines()
            report_samples = [sample.strip() for sample in report_samples_name]
            samples = list(zip(image_samples, report_samples))
            return samples
        else:
            patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
            print(f"start prepraring samples")
            with Pool() as pool:
                # Use a lambda or partial to pass additional arguments
                results = pool.starmap(self.process_patient_folder, [(folder, self.accession_to_text, self.paths) for folder in patient_folders])

            # Combine all patient folder results
            samples = [item for sublist in results for item in sublist]
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
    def nii_img_to_tensor(self, path, transform):
        img_data = np.load(path)['arr_0']
        img_data= np.transpose(img_data, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (img_data / 1000).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
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

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor

    # original function to convert nii file to tensor, in nii.gz format
    # def nii_img_to_tensor(self, path, transform):
    #     nii_img = nib.load(str(path))
    #     img_data = nii_img.get_fdata()

    #     # todo: change this line to be auto config
    #     # df = pd.read_csv("/mnt/input/CT-RATE/organized_dataset/csv_dir/metadata/train_metadata.csv") #select the metadata
    #     df = self.metadata

    #     # todo: maybe the resize process in the function is not needed, see if remove is reasonable

    #     file_name = path.split("/")[-1]
    #     row = df[df['VolumeName'] == file_name]
    #     slope = float(row["RescaleSlope"].iloc[0])
    #     intercept = float(row["RescaleIntercept"].iloc[0])
    #     xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    #     z_spacing = float(row["ZSpacing"].iloc[0])

    #     # Define the target spacing values
    #     target_x_spacing = 0.75
    #     target_y_spacing = 0.75
    #     target_z_spacing = 1.5

    #     current = (z_spacing, xy_spacing, xy_spacing)
    #     target = (target_z_spacing, target_x_spacing, target_y_spacing)

    #     img_data = slope * img_data + intercept

    #     img_data = img_data.transpose(2, 0, 1)

    #     tensor = torch.tensor(img_data)
    #     tensor = tensor.unsqueeze(0).unsqueeze(0)

    #     img_data = resize_array(tensor, current, target)
    #     img_data = img_data[0][0]
    #     img_data= np.transpose(img_data, (1, 2, 0))

    #     hu_min, hu_max = -1000, 1000
    #     img_data = np.clip(img_data, hu_min, hu_max)

    #     img_data = (((img_data ) / 1000)).astype(np.float32)
    #     slices=[]

    #     tensor = torch.tensor(img_data)
    #     # Get the dimensions of the input tensor
    #     target_shape = (480,480,240)

    #     # Extract dimensions
    #     h, w, d = tensor.shape

    #     # Calculate cropping/padding values for height, width, and depth
    #     dh, dw, dd = target_shape
    #     h_start = max((h - dh) // 2, 0)
    #     h_end = min(h_start + dh, h)
    #     w_start = max((w - dw) // 2, 0)
    #     w_end = min(w_start + dw, w)
    #     d_start = max((d - dd) // 2, 0)
    #     d_end = min(d_start + dd, d)

    #     # Crop or pad the tensor
    #     tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    #     pad_h_before = (dh - tensor.size(0)) // 2
    #     pad_h_after = dh - tensor.size(0) - pad_h_before

    #     pad_w_before = (dw - tensor.size(1)) // 2
    #     pad_w_after = dw - tensor.size(1) - pad_w_before

    #     pad_d_before = (dd - tensor.size(2)) // 2
    #     pad_d_after = dd - tensor.size(2) - pad_d_before

    #     tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    #     tensor = tensor.permute(2, 0, 1)

    #     tensor = tensor.unsqueeze(0)

    #     return tensor


    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text