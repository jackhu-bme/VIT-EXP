import os
from multiprocessing import Pool
import shutil

def get_copy_pairs(src, dst):
    copy_pairs = []
    print("start to get copy pairs")
    for root, dirs, files in os.walk(src):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = src_file.replace(src, dst)
            dst_dir = os.path.dirname(dst_file)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            copy_pairs.append((src_file, dst_file))
            print(f"add copy pair: {src_file} to {dst_file}")
            # return copy_pairs
    return copy_pairs


def copy(src, dst):
    # os.system(f"cp -r {src} {dst}")
    shutil.copy(src, dst)
    print(f"finish copy {src} to {dst}")


if __name__ == "__main__":
    # src = "/home/xufluo/data/RadGenome"
    # dst = "/home/xufluo/blobmnt/RadGenome/full_dataset_revised"
    src = "../combined_seg_17_cls_revise"
    dst = "/mnt/input/RadGenome/combined_seg_17_cls_new_revise"
    # /home/xufluo/data/RadGenome/data/train_anatomy_mask_combined
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(dst):
        os.makedirs(dst)
    # cpu count
    cpu_n = os.cpu_count()
    pool = Pool(cpu_n)
    copy_pairs = get_copy_pairs(src, dst)
    pool.starmap(copy, copy_pairs)
    pool.close()
    pool.join()


