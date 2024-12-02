import torch
from transformer_maskgit import CTViT, CTViT3D
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInferenceFast
import accelerate

import argparse

import yaml

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


def main(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
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

    # clip_model_path_for_infer = "../output_train_scratch/train_from_scratch_vit_hpc_1_dim_384/2024-11-16_06-57-22/checkpoints/CTClip.100000.pt"
    clip_model_path_for_infer = args.model_path
    clip.load(clip_model_path_for_infer, check=True)


    inference = CTClipInferenceFast(
        clip,
        data_folder = '/mnt/input/CT-RATE/organized_dataset/val_images_preprocessed',
        reports_file= "/mnt/input/CT-RATE/organized_dataset/csv_dir/reports/validation_reports.csv",
        labels = "/mnt/input/CT-RATE/organized_dataset/csv_dir/labels/valid_predicted_labels.csv",
        batch_size = 1,
        results_folder=args.results_folder,
        num_train_steps = 1,
    )
    inference.infer()


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    # required arguments
    args.add_argument("--config", type=str, required=True)
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--results_folder", type=str, required=True)

    # args.add_argument("--config", type=str, default="../configs/train_from_scratch/ct_clip_vit_hpc_v3_1.yaml")
    # args.add_argument("--model_path", type=str, default="../output_train_scratch/train_from_scratch_vit_hpc_1_dim_384/2024-11-16_06-57-22/checkpoints/CTClip.100000.pt")
    # args.add_argument("--results_folder", type=str, default="../results_inference_zeroshot_ctvit/")


    args = args.parse_args()

    main(args)
    
