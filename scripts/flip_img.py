import os

import pandas as pd

import numpy as np

import torch

import torch.nn.functional as F

import argparse


from multiprocessing import Pool

train_mask_combined_dir = "/mnt/input/RadGenome/combined_seg_17_cls"

save_revise_train_mask_dir = "../combined_seg_17_cls_revise"
os.makedirs(save_revise_train_mask_dir, exist_ok=True)

# load train metadata
metadata_path = "/mnt/input/CT-RATE/organized_dataset/csv_dir/metadata/train_metadata.csv"

df = pd.read_csv(metadata_path)


# for root, dirs, files in os.walk(train_mask_combined_dir):
#     for file in files:
# for file in os.listdir(train_mask_combined_dir):
def process(file):
    if file.endswith(".npz"):
        file_name = file.replace(".npz", ".nii.gz")
        row = df[df['VolumeName'] == file_name]
        # slope = float(row["RescaleSlope"].iloc[0])
        # intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
        n_slices = int(row["NumberofSlices"].iloc[0])
        n_rows = int(row["Rows"].iloc[0])
        n_cols = int(row["Columns"].iloc[0])
        print(f"File: {file_name}, XY Spacing: {xy_spacing}, Z Spacing: {z_spacing}, Number of Slices: {n_slices}, Rows: {n_rows}, Columns: {n_cols}")
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        target_n_slices = int(n_slices * z_spacing / target_z_spacing)
        target_n_rows = int(n_rows * xy_spacing / target_y_spacing)
        target_n_cols = int(n_cols * xy_spacing / target_x_spacing)
        print(f"Target Number of Slices: {target_n_slices}, Rows: {target_n_rows}, Columns: {target_n_cols}")
        # load the npz
        npz_file = os.path.join(train_mask_combined_dir, file)
        npz_data = np.load(npz_file, allow_pickle=True)
        print(f"npz data shape: {npz_data['arr_0'].shape}")
        # flip on dim 1 and 2
        flipped_data = np.flip(npz_data["arr_0"], axis=(1, 2))
        # current shape: (n_c, x, y, z)
        # resize to target shape
        resized_data = torch.tensor(flipped_data.copy()).unsqueeze(0).int().float().cuda()
        # interpolate mask data, use area interpolation
        resized_data = F.interpolate(resized_data, size=(target_n_rows, target_n_cols, target_n_slices), mode='nearest')
        # convert to 0, 1
        resized_data = torch.ceil(resized_data).int().cpu().numpy()
        resized_data = resized_data.squeeze().astype(bool)
        print(f"resized data shape: {resized_data.shape}")
        # save as npz
        save_path = os.path.join(save_revise_train_mask_dir, file)
        # np.savez(save_path, resized_data)
        np.savez_compressed(save_path, resized_data)
        torch.cuda.empty_cache()

        # exit()
            
if __name__ == "__main__":
    # with Pool(4) as p:
    #     p.map(process, os.listdir(train_mask_combined_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_split', type=int, default=0)
    
    args = parser.parse_args()


    all_file_list = os.listdir(train_mask_combined_dir)
    n_splits = 4
    current_split = args.current_split

    current_split_list = all_file_list[current_split::n_splits]

    # for file in current_split_list:
    #     process(file)
    with Pool(4) as p:
        p.map(process, current_split_list)















