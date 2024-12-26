import os
import numpy as np
from multiprocessing import Pool, cpu_count

def load_npz_file(args):
    filename, directory = args
    file_path = os.path.join(directory, filename)
    try:
        a = np.load(file_path)["arr_0"]
        print(f"successfully load {file_path}")
        return None
    except Exception as e:
        print(f"errors: {e}")
        return (filename, str(e))

def find_npz_files_walk(directory):
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append((file, root))
                print(file)
    return npz_files

def find_npz_load_errors_walk_multiprocess(directory):
    error_files = []
    npz_files = find_npz_files_walk(directory)
    with Pool(cpu_count()) as pool:
        results = pool.map(load_npz_file, npz_files)
    for result in results:
        if result is not None:
            error_files.append(result)
    return error_files

# Example usage:
# errors = find_npz_load_errors_walk_multiprocess('/home/xufluo/blobmnt/CT-RATE/sub_dataset/prepocessed_float32_divide_5/divide_0')
# print(errors)

npz_folder_path = "/mnt/input/RadGenome/combined_seg_17_cls_new"
errors = find_npz_load_errors_walk_multiprocess(npz_folder_path)
with open("error_npz_seg_cls.txt", "w") as f:
    for error in errors:
        f.write(str(error) + "\n")



# import os
# import numpy as np
# from multiprocessing import Pool, cpu_count

# def load_npz_file(args):
#     filename, directory = args
#     file_path = os.path.join(directory, filename)
#     try:
#         a = np.load(file_path)
#         print(f"succeefully load {file_path}")
#         return None
#     except Exception as e:
#         return (filename, str(e))

# def find_npz_load_errors_walk(directory):
#     error_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.npz'):
#                 file_path = os.path.join(root, file)
#                 result = load_npz_file((file, root))
#                 if result is not None:
#                     error_files.append(result)
#     return error_files

# # Example usage:
# errors = find_npz_load_errors_walk('')
# print(errors)

