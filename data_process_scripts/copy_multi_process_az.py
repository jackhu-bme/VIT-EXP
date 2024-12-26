import os
from multiprocessing import Pool
import shutil

def get_copy_pairs(src, dst, file_list=None):
    if file_list is not None:
        with open(file_list, "r") as f:
            error_files = f.readlines()
        error_files = [file.strip() for file in error_files]
    copy_pairs = []
    print("start to get copy pairs")
    for root, dirs, files in os.walk(src):
        for file in files:
            if file_list is not None and not file in error_files:
                continue
            print(f"start to process {file}")
            src_file = os.path.join(root, file)
            dst_file = src_file.replace(src, dst)
            dst_dir = os.path.dirname(dst_file)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            copy_pairs.append((src_file, dst_file))
            print(f"add copy pair: {src_file} to {dst_file}")
            # return copy_pairs
    return copy_pairs

# read the token from environment variable

az_token = os.environ.get("AZ_TOKEN")

position = "https://xufluo.blob.core.windows.net/medical-images/"

def copy(src, dst):
    # os.system(f"cp -r {src} {dst}")
    # shutil.copy(src, dst)
    # print(f"finish copy {src} to {dst}")
    # use azcopy, copy from local src to azure blob dst
    print(f"start to copy {src} to {dst}")
    res = os.popen(f"azcopy copy '{src}' '{position}{dst}{az_token}'").read()
    print(res)


if __name__ == "__main__":
    # src = "/home/xufluo/data/LUNA16/unzip_data"
    # dst = "/home/xufluo/blobmnt/LUNA16/unzip_data"
    src = "../combined_seg_17_cls_revise"
    dst = "RadGenome/combined_seg_17_cls_new_revise"
    file_list = "./errors.txt"
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    copy_pairs = get_copy_pairs(src, dst) #, file_list)
    print(f"n copy pairs: {len(copy_pairs)}")
    # exit()
    # for src_file, dst_file in copy_pairs:
    #     copy(src_file, dst_file)
    #     exit()
    pool = Pool(24)
    pool.starmap(copy, copy_pairs)
    pool.close()
    pool.join()


