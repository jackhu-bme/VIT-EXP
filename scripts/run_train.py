from transformer_maskgit import CTViT, CTViT3D
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer

from datetime import timedelta

import yaml

import argparse

import torch
import random
import numpy as np

import os


import wandb

import time

from accelerate import Accelerator, DistributedDataParallelKwargs

from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs
# ProjectConfiguration


def create_img_encoder(config):
    if config.get("use_seg", False):
        seg_config = config.get("seg_head", {})
    else:
        seg_config = {}
    if config.get("arch_name") == "CTViT3D":
        image_encoder = CTViT3D(
            # dim = 512,
            dim = config.get("dim", 768),
            # codebook_size = 8192,
            image_size = config.get("image_size", 480),
            patch_size = config.get("patch_size", 20),
            temporal_size= config.get("temporal_size", 240),
            temporal_patch_size= config.get("temporal_patch_size", 10),
            transformer_blocks = config.get("transformer_blocks", 8),
            dim_head = config.get("dim_head", 32),
            heads = config.get("heads", 8),
            use_flash_attention = config.get("use_flash_attention", True),
            use_seg = config.get("use_seg", False),
            seg_head_n_layers = seg_config.get("n_layers", 2),
            seg_head_layer_type = seg_config.get("layer_type", "mlp"),
            seg_head_in_dim = seg_config.get("in_dim", 256),
            seg_head_mid_dim = seg_config.get("mid_dim", 128),
            seg_head_out_dim = seg_config.get("out_dim", 22), # 22 classes for segmentation in TotalSegmentor
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
    return image_encoder



def main(config, args):

    project_name = config.get("project_name", "CT-CLIP-EXP")
    exp_name = config.get("exp_name", "train_from_scratch_default")
    # current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    exp_folder = os.path.join(config["results_folder"], exp_name) #, current_time)
    ckpt_folder = os.path.join(exp_folder, "checkpoints")
    os.makedirs(ckpt_folder, exist_ok=True)

    wandb_folder = os.path.join(exp_folder, "wandb")
    os.makedirs(wandb_folder, exist_ok=True)

    wandb_mode = "offline" if args.debug else "online"

    project_config = ProjectConfiguration(
        project_dir=exp_folder,        
        automatic_checkpoint_naming=True,  
        total_limit=10000            
        )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], 
                              log_with="wandb", 
                              project_config=project_config, 
                              gradient_accumulation_steps=config["trainer"].get("gradient_accumulation_steps", 1))


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


    # fix the random seed based on the config args
    seed = int(config["random_seed"])

    print(f"Setting random seed to {seed}")


    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    # print("---------")
    # print(tokenizer.pad_token_id)
    # print(tokenizer.mask_token_id)
    # print("-----------")

    image_encoder = create_img_encoder(config["arch"])
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
        use_all_token_embeds = False,
        config=config["ct_clip_arch"],
    )

    # if resume_path is not None:
    #     print(f"Resuming state dict from checkpoint: {resume_path}")
    #     clip.load(resume_path)

    # also resume the trainer

    # now try to define the accelerator within the run_trian python script!


    trainer = CTClipTrainer(
        clip,
        config=config,
        tokenizer=tokenizer,
        accelerator=accelerator,
        # reports_file_train= config["reports_file_train"],
        # reports_file_valid= config["reports_file_valid"],
        # metadata_train= config["metadata_train"],
        # data_train= config["data_train"],
        # data_valid = config["data_valid"],
        # use_seg = config.get("use_seg", False),
        # seg_data_train = config.get("seg_data_train", None),
        # seg_data_valid = config.get("seg_data_valid", None),
        # seg_mask_train = config.get("seg_mask_train", None),
        # seg_mask_valid = config.get("seg_mask_valid", None),
        # balance_report_seg = config.get("balance_report_seg", 1.0),
        # labels = config["labels"],
        # batch_size = config["batch_size"],
        results_folder = ckpt_folder,
        auto_resume = args.auto_resume,
        # # results_folder = config["results_folder"],
        # num_train_steps = config["num_train_steps"],
        # num_workers = config["num_workers"],
        # accelerate_kwargs = {"gradient_accumulation_steps": config["gradient_accumulation_steps"]},
        resume_path = resume_path,
        )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    
    # read the config and set the args
    args = argparse.ArgumentParser(description='CT-CLIP')
    args.add_argument('--config', required=True, help='path to the config file')
    args.add_argument('--auto_resume', action='store_true', help='auto resume from the latest checkpoint')
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




