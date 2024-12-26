import os
import numpy as np

from multiprocessing import Pool

ori_wrong_mask_dir = "../combined_seg_17_cls_revise"

save_correct_mask_dir = "../train_combined_seg_17_cls_final"
os.makedirs(save_correct_mask_dir, exist_ok=True)

def process(mask_name):
    if mask_name.endswith(".npz"):
        mask_file = os.path.join(ori_wrong_mask_dir, mask_name)
        mask_data = np.load(mask_file, allow_pickle=True)["arr_0"]
        mask_data = mask_data.transpose((0, 3, 1, 2))
        save_path = os.path.join(save_correct_mask_dir, mask_name)
        np.savez_compressed(save_path, mask_data)
        print(f"Saved {save_path}")

# for mask_name in sorted(os.listdir(ori_wrong_mask_dir)):
#     if mask_name.endswith(".npz"):
#         mask_file = os.path.join(ori_wrong_mask_dir, mask_name)
#         mask_data = np.load(mask_file, allow_pickle=True)["arr_0"]
#         mask_data = mask_data.transpose((0, 3, 1, 2))
#         save_path = os.path.join(save_correct_mask_dir, mask_name)
#         np.savez_compressed(save_path, mask_data)
#         print(f"Saved {save_path}")
        # exit() # for test

if __name__ == "__main__":
    mask_name_list = sorted(os.listdir(ori_wrong_mask_dir))
    n_process = os.cpu_count()
    with Pool(n_process) as pool:
        pool.map(process, mask_name_list)
    print("All done")


