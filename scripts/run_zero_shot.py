import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


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

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 294912,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)



inference = CTClipInference(
    clip,
    data_folder = '/mnt/input/CT-RATE/organized_dataset/val_images_preprocessed',
    reports_file= "/mnt/input/CT-RATE/organized_dataset/csv_dir/reports/validation_reports.csv",
    labels = "/mnt/input/CT-RATE/organized_dataset/csv_dir/labels/valid_predicted_labels.csv",
    batch_size = 1,
    results_folder="../results_inference_zeroshot_demo/",
    num_train_steps = 1,
)

clip_model_path_for_infer = "/mnt/input/CT-CLIP/ckpt/split_1_sub_2/CTClip.107000.pt"

inference.load_model(clip_model_path_for_infer)

inference.infer()
