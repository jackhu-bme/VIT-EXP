import os

import numpy as np

from multiprocessing import Pool

import torch.nn.functional as F

import torch

# determine the mask for each img file, search for the corresponding mask file

# and check the shape of mask and image, is the mask the same size as the image? (last 3 dimensions)                     

ori_train_dir = "/mnt/input/CT-RATE/data_volumes/dataset/train_preprocessed_float32"

mask_bank_dir = "../combined_seg_17_cls_revise"

# names for train files: 
# train_2_a_2.npz for instance, the name format: train_{patient_id}_{scan_id}_{reconstruction_id}.npz

# names for mask files:
# train_2_a_1.npz for instance, the name format: train_{patient_id}_{scan_id}_1.npz

# for different reconstruction_id, the mask file name is the same, so should alwats search the mask file with reconstruction_id = 1 for each training image file

save_mask_selected_dir = "./train_mask_final"


def select_mask_file(train_img_file_path):
    # get the patient_id, scan_id, reconstruction_id from the file name
    file_name = os.path.basename(train_img_file_path)
    parts = file_name.split("_")
    patient_id = parts[1]
    scan_id = parts[2]
    reconstruction_id = parts[3].split(".")[0]
    mask_file_name = f"train_{patient_id}_{scan_id}_1.npz"
    mask_file_path = os.path.join(mask_bank_dir, mask_file_name)
    return mask_file_path

def get_train_img_path_list(train_img_dir):
    # walk, and get full abs path list
    train_img_path_list = []
    tmp_path_cache_list = os.path.join(train_img_dir, "cache_data_list", "image_samples.txt")
    if os.path.exists(tmp_path_cache_list):
        with open(tmp_path_cache_list, "r") as f:
            for line in f.readlines():
                train_img_path_list.append(line.strip())
        return train_img_path_list
    for root, dirs, files in os.walk(train_img_dir):
        for file in files:
            if file.endswith(".npz"):
                print(f"Add {file} to the list")
                train_img_path_list.append(os.path.join(root, file))
    return train_img_path_list

def select_compare_save_single(train_img_path, save_mask_selected_dir):
        mask_file_path = select_mask_file(train_img_path)
        # load the mask and img
        mask_data = np.load(mask_file_path, allow_pickle=True)["arr_0"].transpose((0, 3, 1, 2))
        img_data = np.load(train_img_path, allow_pickle=True)["arr_0"]
        if mask_data.shape[-3:] != img_data.shape[-3:]:
            print(f"Error: mask shape {mask_data.shape} is not the same as img shape {img_data.shape}")
            # try resize to the same shape using pytorch
            # mask_data = F.interpolate(torch.tensor(mask_data), size=img_data.shape[-3:], mode="trilinear", align_corners=False).numpy()
            # resize using gpu
            mask_data = mask_data.astype(np.float32)
            mask_data = F.interpolate(torch.tensor(mask_data).unsqueeze(0), size=img_data.shape[-3:], mode="trilinear", align_corners=False).squeeze().numpy()
            mask_data = mask_data.astype(bool)
        save_mask_path = os.path.join(save_mask_selected_dir, os.path.basename(mask_file_path))
        np.savez_compressed(save_mask_path, mask_data)
        print(f"Save mask to {save_mask_path}")


# def select_compare_save(train_img_path_list, save_mask_selected_dir):
    
    # for train_img_path in train_img_path_list:
    #     mask_file_path = select_mask_file(train_img_path)
    #     # load the mask and img
    #     mask_data = np.load(mask_file_path, allow_pickle=True)["arr_0"].transpose((0, 3, 1, 2))
    #     img_data = np.load(train_img_path, allow_pickle=True)["arr_0"]
    #     if mask_data.shape[-3:] != img_data.shape[-3:]:
    #         print(f"Error: mask shape {mask_data.shape} is not the same as img shape {img_data.shape}")
    #         # try resize to the same shape using pytorch
    #         # mask_data = F.interpolate(torch.tensor(mask_data), size=img_data.shape[-3:], mode="trilinear", align_corners=False).numpy()
    #         # resize using gpu
    #         mask_data = mask_data.astype(np.float32)
    #         mask_data = F.interpolate(torch.tensor(mask_data).unsqueeze(0), size=img_data.shape[-3:], mode="trilinear", align_corners=False).squeeze().numpy()
    #         mask_data = mask_data.astype(bool)
    #         print(f"current mask shape {mask_data.shape}, img shape {img_data.shape}")
    #     save_mask_path = os.path.join(save_mask_selected_dir, os.path.basename(mask_file_path))
    #     # np.savez(save_mask_path, mask_data)
    #     np.savez_compressed(save_mask_path, mask_data)
    #     print(f"Save mask to {save_mask_path}")
        # exit() # for test
    
    
    


if __name__ == "__main__":
    train_img_path_list = get_train_img_path_list(ori_train_dir)
    os.makedirs(save_mask_selected_dir, exist_ok=True)
    # select_compare_save(train_img_path_list, save_mask_selected_dir)
    # multiprocess
    n_process = os.cpu_count()
    with Pool(n_process) as pool:
        pool.starmap(select_compare_save_single, [(train_img_path, save_mask_selected_dir) for train_img_path in train_img_path_list])






