from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer

import yaml

import argparse

import torch
import random
import numpy as np

import os



def main(config, args):
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


    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 294912,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False

    )
    trainer = CTClipTrainer(
        clip,
        reports_file_train= "/mnt/input/CT-RATE/organized_dataset/csv_dir/reports/train_reports.csv",
        reports_file_valid= "/mnt/input/CT-RATE/organized_dataset/csv_dir/reports/validation_reports.csv",
        data_train= "/mnt/input/CT-RATE/data_volumes/dataset/train",
        data_valid = "/mnt/input/CT-RATE/organized_dataset/val_images_preprocessed",
        labels = "/mnt/input/CT-RATE/organized_dataset/csv_dir/labels/train_predicted_labels.csv",
        batch_size = 2,
        results_folder="../output_train_scratch_resume",
        num_train_steps = 200002,
        num_workers = 16,
        accelerate_kwargs = {"gradient_accumulation_steps":2},
        resume_path = args.resume
        # resume_path="../ckpts/CTClip.60500.pt"
    )

    trainer.train()




if __name__ == "__main__":
    
    # read the config and set the args
    args = argparse.ArgumentParser(description='CT-CLIP')
    args.add_argument('--config', required=True, help='path to the config file')
    args.add_argument('--resume', required=True, help='path to the resume file')
    args = args.parse_args()

    config_path = os.path.join("configs/train_from_scratch", args.config)

    print(f"loading config path: {config_path}")

    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    main(config, args)




