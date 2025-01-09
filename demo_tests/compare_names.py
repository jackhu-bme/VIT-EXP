import os

data_dir = "/mnt/input/RadGenome/valid_preprocessed_img_organized_fp16"
mask_dir = "/mnt/input/RadGenome/valid_preprocessed_mask_combined_bool"

def get_file_names(data_dir, mask_dir):
    data_files = os.listdir(data_dir)
    mask_files = os.listdir(mask_dir)
    return data_files, mask_files


def compare_names(data_files, mask_files):
    data_files = set(data_files)
    mask_files = set(mask_files)
    diff_1 = data_files - mask_files
    diff_2 = mask_files - data_files
    return diff_1, diff_2


data_files, mask_files = get_file_names(data_dir, mask_dir)
diff_1, diff_2 = compare_names(data_files, mask_files)
print(diff_1)
print(diff_2)

print(f"len data_files: {len(data_files)}")
print(f"len mask_files: {len(mask_files)}")
print(f"len diff_1: {len(diff_1)}")
print(f"len diff_2: {len(diff_2)}")






