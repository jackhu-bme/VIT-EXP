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
import tqdm

from multiprocessing import Pool

import pickle


class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.cache_data_list_folder = os.path.join(data_folder, './tmp_cache_data_list_val')
        os.makedirs(self.cache_data_list_folder, exist_ok=True)
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    # def prepare_samples(self):
    #     samples = []
    #     patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

    #     # Read labels once outside the loop
    #     test_df = pd.read_csv(self.labels)
    #     test_label_cols = list(test_df.columns[1:])
    #     test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

    #     for patient_folder in tqdm.tqdm(patient_folders):
    #         accession_folders = glob.glob(os.path.join(patient_folder, '*'))

    #         for accession_folder in accession_folders:
    #             nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))

    #             for nii_file in nii_files:
    #                 accession_number = nii_file.split("/")[-1]

    #                 accession_number = accession_number.replace(".npz", ".nii.gz")
    #                 if accession_number not in self.accession_to_text:
    #                     continue

    #                 impression_text = self.accession_to_text[accession_number]
    #                 text_final = ""
    #                 for text in list(impression_text):
    #                     text = str(text)
    #                     if text == "Not given.":
    #                         text = ""

    #                     text_final = text_final + text

    #                 onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values

    #                 print(f"onehotlabels: {onehotlabels}")

    #                 if len(onehotlabels) > 0:
    #                     samples.append((nii_file, text_final, onehotlabels[0]))
    #                     self.paths.append(nii_file)
    #     return samples

    def process_patient_folder(self, patient_folder, accession_to_text, paths, test_df):
        samples = []
        accession_folders = glob.glob(os.path.join(patient_folder, '*'))
        for accession_folder in accession_folders:
            nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))
            # print(f"nii files: {nii_files}, accession folder:{accession_folder}")
            for nii_file in nii_files:
                accession_number = nii_file.split("/")[-1].replace(".npz", ".nii.gz")
                if accession_number not in accession_to_text:
                    continue

                impression_text = self.accession_to_text[accession_number]
                text_final = ""
                for text in list(impression_text):
                    text = str(text)
                    if text == "Not given.":
                        text = ""

                    text_final = text_final + text

                onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values

                # print(f"onehotlabels: {onehotlabels}")

                if len(onehotlabels) > 0:
                    samples.append((nii_file, text_final, onehotlabels[0]))
                    paths.append(nii_file)

                # input_text_concat = "".join(str(text) for text in impression_text) if impression_text else ""
                # samples.append((nii_file, input_text_concat))
                # # print(f"appending sample:{nii_file}")
                # paths.append(nii_file)
        return samples

    def prepare_samples(self):
        print(f"debugp: cache_data_list_folder: {self.cache_data_list_folder}")
        if os.path.exists(os.path.join(self.cache_data_list_folder, 'image_samples.txt')) and \
            os.path.exists(os.path.join(self.cache_data_list_folder, 'report_samples.txt')) and \
            os.path.exists(os.path.join(self.cache_data_list_folder, 'label_samples.pkl')):
            with open(os.path.join(self.cache_data_list_folder, 'image_samples.txt'), 'r') as f:
                image_samples_name = f.readlines()
            image_samples = [sample.strip() for sample in image_samples_name]
            with open(os.path.join(self.cache_data_list_folder, 'report_samples.txt'), 'r') as f:
                report_samples_name = f.readlines()
            report_samples = [sample.strip() for sample in report_samples_name]
            with open(os.path.join(self.cache_data_list_folder, 'label_samples.pkl'), 'rb') as f:
                label_samples = pickle.load(f)
            samples = list(zip(image_samples, report_samples, label_samples))
            print(f"finished preparing samples with cache txt in ct report infer, the number of samples: {len(samples)}")
            return samples
        else:
            # Read labels once outside the loop
            test_df = pd.read_csv(self.labels)
            test_label_cols = list(test_df.columns[1:])
            test_df['one_hot_labels'] = list(test_df[test_label_cols].values)
            patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
            print(f"debugp: patient folders length: {len(patient_folders)}, first 5 folder: {patient_folders[0:5]}")
            print(f"start prepraring samples")
            with Pool() as pool:
                # Use a lambda or partial to pass additional arguments
                results = pool.starmap(self.process_patient_folder, [(folder, self.accession_to_text, self.paths, test_df) for folder in patient_folders])

            print(f"debugp results length: {len(results)}, first 5 results: {results[0:5]}")

            # Combine all patient folder results
            samples = [item for sublist in results for item in sublist]
            print(f"finished preparing samples, the number of samples: {len(samples)}")
            # Save the samples to cache
            image_samples = [sample[0] for sample in samples]
            report_samples = [sample[1] for sample in samples]
            label_samples = [sample[2] for sample in samples]
            os.makedirs(self.cache_data_list_folder, exist_ok=True)
            with open(os.path.join(self.cache_data_list_folder, 'image_samples.txt'), 'w') as f:
                for sample in image_samples:
                    f.write(f"{sample}\n")
            with open(os.path.join(self.cache_data_list_folder, 'report_samples.txt'), 'w') as f:
                for sample in report_samples:
                    f.write(f"{sample}\n")
            # write pickle
            with open(os.path.join(self.cache_data_list_folder, 'label_samples.pkl'), 'wb') as f:
                pickle.dump(label_samples, f)
            return samples


    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        img_data = np.load(path)['arr_0']
        img_data= np.transpose(img_data, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (img_data / 1000).astype(np.float32)
        # slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
        # Extract dimensions
        h, w, d = tensor.shape

        if target_shape == (h, w, d):
            pass
        else:
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

    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        name_acc = nii_file.split("/")[-2]
        return video_tensor, input_text, onehotlabels, name_acc
