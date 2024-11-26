import torch
from transformer_maskgit import CTViT, CTViT3D
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


import yaml
config = yaml.load(open("../configs/train_from_scratch/ct_clip_vit_hpc_v3_1.yaml", "r"), Loader=yaml.FullLoader)

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

# image_encoder = CTViT(
#     dim = 512,
#     codebook_size = 8192,
#     image_size = 480,
#     patch_size = 20,
#     temporal_patch_size = 10,
#     spatial_depth = 4,
#     temporal_depth = 4,
#     dim_head = 32,
#     heads = 8
# )

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

clip_model_path_for_infer = "/mnt/input/CT-CLIP-VIT/train_from_scratch_vit_hpc_1_dim_384_div0/2024-11-22_05-51-07/checkpoints/CTClip.40000.pt"
clip.load(clip_model_path_for_infer, check=True)


inference = CTClipInference(
    clip,
    data_folder = '/mnt/input/CT-RATE/organized_dataset/val_images_preprocessed',
    reports_file= "/mnt/input/CT-RATE/organized_dataset/csv_dir/reports/validation_reports.csv",
    labels = "/mnt/input/CT-RATE/organized_dataset/csv_dir/labels/valid_predicted_labels.csv",
    batch_size = 1,
    results_folder="../results_inference_zeroshot_ctvit_full_data/",
    num_train_steps = 1,
)



inference.infer()
