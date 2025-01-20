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
    arch_config = config["arch"]
    arch_name = arch_config.get("arch_name", "CTViT3D")
    seg_config = arch_config.get("seg_head", None)
    seg_kwargs = {}
    if seg_config is None:
        seg_kwargs["use_seg"] = False
        # use_seg = False
        # raise NotImplementedError # todo: deal with this later
    else:
        # use_seg = True
        seg_kwargs["use_seg"] = True
        seg_kwargs["seg_head_n_layers"] = seg_config.get("n_layers", 2)
        seg_kwargs["seg_head_layer_type"] = seg_config.get("layer_type", "mlp")
        seg_kwargs["seg_head_in_dim"] = seg_config.get("in_dim", 256)
        seg_kwargs["seg_head_mid_dim"] = seg_config.get("mid_dim", 128)
        seg_kwargs["seg_head_out_dim"] = seg_config.get("out_dim", 22) # 22 classes for segmentation in TotalSegmentor
    print(f"arch config: {arch_config}")
    # exit()
    if arch_name == "CTViT3D":
        image_encoder = CTViT3D(
                    # dim = 512,
                    dim = arch_config.get("dim", 768),
                    # codebook_size = 8192,
                    image_size = arch_config.get("image_size", 480),
                    patch_size = arch_config.get("patch_size", 20),
                    temporal_size= arch_config.get("temporal_size", 240),
                    temporal_patch_size = arch_config.get("temporal_patch_size", 10),
                    transformer_blocks = arch_config.get("transformer_blocks", 8),
                    dim_head = arch_config.get("dim_head", 32),
                    heads = arch_config.get("heads", 8),
                    use_flash_attention = arch_config.get("use_flash_attention", True),
                    **seg_kwargs,
                    # use_seg = use_seg,
                    # seg_head_n_layers = seg_config.get("n_layers", 2),
                    # seg_head_layer_type = seg_config.get("layer_type", "mlp"),
                    # seg_head_in_dim = seg_config.get("in_dim", 256),
                    # seg_head_mid_dim = seg_config.get("mid_dim", 128),
                    # seg_head_out_dim = seg_config.get("out_dim", 22), # 22 classes for segmentation in TotalSegmentor
                )
    else:
        return NotImplementedError

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
    
