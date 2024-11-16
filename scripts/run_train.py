from transformer_maskgit import CTViT, CTViT3D
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer

import yaml

import argparse

import torch
import random
import numpy as np

import os

import wandb

import time

from accelerate import Accelerator



def main(config, args):

    project_name = config.get("project_name", "CT-CLIP-EXP")
    exp_name = config.get("exp_name", "train_from_scratch_default")
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    exp_folder = os.path.join(config["results_folder"], exp_name, current_time)
    ckpt_folder = os.path.join(exp_folder, "checkpoints")
    os.makedirs(ckpt_folder, exist_ok=True)

    wandb_folder = os.path.join(exp_folder, "wandb")
    os.makedirs(wandb_folder, exist_ok=True)

    wandb_mode = "offline" if args.debug else "online"

    # wandb_logger = wandb.init(project=project_name, name=exp_name, config=config, mode=wandb_mode, dir=wandb_folder)

    accelerator = Accelerator(log_with="wandb")

    accelerator.init_trackers(
        project_name = project_name,
        init_kwargs = {"wandb": {
            "name": exp_name,
            "mode": wandb_mode,
            "dir": wandb_folder,
            "config": config
        }
        }
    )


    txt_folder = os.path.join(exp_folder, "txt")
    os.makedirs(txt_folder, exist_ok=True)

    # write the output of git status and git log the file
    os.system("git status > " + os.path.join(txt_folder, "git_status.txt"))
    os.system("git log > " + os.path.join(txt_folder, "git_log.txt"))
    #copy to wandb folder
    os.system("cp " + os.path.join(txt_folder, "git_status.txt") + " " + os.path.join(wandb_folder, "git_status.txt"))
    os.system("cp " + os.path.join(txt_folder, "git_log.txt") + " " + os.path.join(wandb_folder, "git_log.txt"))

    # save the txt folder to wandb
    # wandb.save(os.path.join(wandb_folder, "git_status.txt"))
    # wandb.save(os.path.join(wandb_folder, "git_log.txt"))

    wandb_logger = accelerator.get_tracker("wandb")

    # wandb.save(os.path.join(wandb_folder, "git_status.txt"))
    # wandb.save(os.path.join(wandb_folder, "git_log.txt"))

    # fix the random seed based on the config args
    # 设置随机种子
    seed = int(config["random_seed"])

    print(f"Setting random seed to {seed}")


    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子（如果使用 GPU）
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，设置所有 GPU 的随机种子

    # 设置 Python 和 NumPy 的随机种子
    random.seed(seed)
    np.random.seed(seed)


    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    print("---------")
    print(tokenizer.pad_token_id)
    print(tokenizer.mask_token_id)
    print("-----------")

    if config.get("arch_name") == "CTViT3D":
        image_encoder = CTViT3D(
            # dim = 512,
            dim = config.get("CTViT3D_dim", 768),
            # codebook_size = 8192,
            image_size = config.get("image_size", 480),
            patch_size = config.get("patch_size", 20),
            temporal_size= config.get("temporal_size", 240),
            temporal_patch_size= config.get("temporal_patch_size", 10),
            transformer_blocks = config.get("transformer_blocks", 8),
            dim_head = config.get("dim_head", 32),
            heads = config.get("heads", 8),
            use_flash_attention = config.get("use_flash_attention", True),
        )
    else:
        image_encoder = CTViT(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )
    #dim_image = 131072,

    resume_path = args.resume if args.resume else None

    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 442368,
        dim_latent = 768,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False

    )

    if resume_path is not None:
        print(f"Resuming state dict from checkpoint: {resume_path}")
        clip.load(resume_path)

    # also resume the trainer
    trainer = CTClipTrainer(
        clip,
        reports_file_train= config["reports_file_train"],
        reports_file_valid= config["reports_file_valid"],
        metadata_train= config["metadata_train"],
        data_train= config["data_train"],
        data_valid = config["data_valid"],
        labels = config["labels"],
        batch_size = config["batch_size"],
        results_folder = ckpt_folder,
        # results_folder = config["results_folder"],
        num_train_steps = config["num_train_steps"],
        num_workers = config["num_workers"],
        accelerate_kwargs = {"gradient_accumulation_steps": config["gradient_accumulation_steps"]},
        wandb_logger = wandb_logger,
        resume_path = resume_path,
        )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    
    # read the config and set the args
    args = argparse.ArgumentParser(description='CT-CLIP')
    args.add_argument('--config', required=True, help='path to the config file')
    args.add_argument('--resume', default=None, help='path to the checkpoint to resume training')
    args.add_argument('--debug', action='store_true', help='debug mode')
    args = args.parse_args()

    config_path = os.path.join("configs/train_from_scratch", args.config)

    print(f"loading config path: {config_path}")

    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # 启用 Flash Attention
    torch.backends.cuda.enable_flash_sdp(True)

    # 启用数学内核（如果需要）
    torch.backends.cuda.enable_math_sdp(True)

    # 启用内存高效内核（如果需要）
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    main(config, args)




