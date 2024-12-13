# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the model and tokenizer
# model_id = "tiiuae/falcon-7b"  # Replace with your desired model ID
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# # Initialize the model with Flash Attention 2
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,  # Use bfloat16 or float16 for Flash Attention
#     use_flash_attention_2=True,   # Enable Flash Attention 2
# ).to("cuda")  # Move model to GPU

# # Prepare input text
# input_text = "Hello my dog is cute and"
# inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Tokenize and move to GPU

# # Generate output
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_length=100)  # Generate text

# # Decode and print the output
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# from transformers import BertTokenizer, BertModel
# import torch

# # 初始化 tokenizer 和模型
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # 准备输入文本
# text = "Hello, how are you?"
# inputs = tokenizer(text, return_tensors='pt')  # 转换为张量格式

# print(inputs)

# # 前向传播，获取输出
# with torch.no_grad():
#     outputs = model(**inputs)

# # 获取最后一层的隐藏状态
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)  # torch.Size([1, 8, 768])

# from transformers import ViTForImageClassification

# from transformers import ViTConfig, ViTModel

# # Initializing a ViT vit-base-patch16-224 style configuration
# configuration = ViTConfig()

# # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
# model = ViTModel(configuration)

# # Accessing the model configuration
# configuration = model.config


# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float16)
# ...


# from transformers import AutoImageProcessor, ViTModel
# import torch
# from datasets import load_dataset

# dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
# image = dataset["test"]["image"][0]

# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# inputs = image_processor(image, return_tensors="pt")

# input_pixel_values = inputs["pixel_values"]

# print(input_pixel_values.shape)

# with torch.no_grad():
#     outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)

# from transformers import AutoImageProcessor, VideoMAEForPreTraining
# import numpy as np
# import torch

# num_frames = 16
# video = list(np.random.randint(0, 256, (num_frames, 3, 224, 224)))

# image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
# model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

# pixel_values = image_processor(video, return_tensors="pt").pixel_values

# num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
# seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
# bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

# print(pixel_values.shape)

# outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
# # loss = outputs.loss

# print(outputs.logits.shape)
