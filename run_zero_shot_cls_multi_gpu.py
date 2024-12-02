import os

# this is for multi gpu inference, but actually we split the ckpts and run them on single gpu individually

CONFIG="configs/train_from_scratch/ct_clip_vit_hpc_v3_1.yaml"

model_dir = "/mnt/input/CT-CLIP-VIT/train_from_scratch_vit_hpc_1_dim_384/2024-11-16_06-57-22/checkpoints/"

model_list = os.listdir(model_dir)

gaps = 10

# sort according to thee ckpt number

model_list.sort(key=lambda x: int(x.split(".")[1].split(".")[0]))

model_list = model_list[::gaps]

n_models = len(model_list)

total_splits = 1

current_split = 0


# define the start and end index of the models to be run on this split

start_index = current_split * n_models // total_splits

end_index = (current_split + 1) * n_models // total_splits

# get the models to be run on this split

run_models = model_list[start_index:end_index]

current_gpu = 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(current_gpu)

# MODEL="/mnt/input/CT-CLIP-VIT/train_from_scratch_vit_hpc_1_dim_384/2024-11-22_01-06-13/checkpoints/CTClip.200000.pt"

results_dir = "/mnt/input/CT-CLIP-VIT/results_fast_inference_zeroshot_ctvit_full_multi_gpu"

for run_model in run_models:
    MODEL = os.path.join(model_dir, run_model)
    RESULTS=os.path.join(results_dir, run_model.replace(".pt", "")) 
    os.makedirs(RESULTS, exist_ok=True)
    os.system(f"python scripts/run_zero_shot_cls_single_gpu.py --config {CONFIG} --model {MODEL} --results {RESULTS}")