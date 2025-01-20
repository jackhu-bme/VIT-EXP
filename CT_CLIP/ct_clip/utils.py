import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

import cv2

import os


def vis_3d_img_list(img_list, slice_ratio_list=[0.25, 0.5, 0.75], img_name="image"):
    """
    Visualize 3D images by sampling slices along each dimension.
    
    Args:
        img_list (list): List of 3D tensors (in 0-1 range).
        slice_ratio_list (list): List of ratios to slice the image along each dimension.
        img_name (str): The base name for the visualization.
    
    Returns:
        list: A list of WandB Image objects.
    """

    # copy the img lists to cpu

    img_list = [img.cpu() for img in img_list]

    wandb_images = {}

    # Ensure all images in the list have the same shape
    img_shape = img_list[0].shape
    for img in img_list:
        if img.shape != img_shape:
            raise ValueError("All images in img_list must have the same shape.")

    # Stack images along a new 4th dimension
    img_stack = torch.stack(img_list, dim=3)  # Shape: (D, H, W, N) where N is the number of images
    
    # Iterate over each dimension for slicing (0, 1, 2)
    for dim_idx, dim_name in enumerate(["dim_0", "dim_1", "dim_2"]):
        slices = []
        for ratio in slice_ratio_list:
            # Calculate the index of the slice along this dimension
            idx_slice = int(img_stack.shape[dim_idx] * ratio)

            if dim_idx == 0:
                slice_img = img_stack[idx_slice, :, :, :]
            elif dim_idx == 1:
                slice_img = img_stack[:, idx_slice, :, :]
            else:  # dim_idx == 2
                slice_img = img_stack[:, :, idx_slice, :]

            slices.append(slice_img)

        # Concatenate slices in the width dimension (axis=3) to compare different images
        slices = np.stack([slice_img.numpy() for slice_img in slices], axis=-1)  # Concatenate along the 4th dimension
        # print(slices.shape)
        # exit()
        h, w, n_imgs, n_slices = slices.shape

        # slices = slices.reshape(h, w * n_imgs, n_slices)

        slices = slices.transpose(2, 0, 3, 1).reshape(h * n_imgs, w * n_slices)

        # # Plot the slices in a single image
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(slices, cmap='gray')
        # ax.axis('off')
        # plt.close(fig)

        # Add a suffix to the image name based on the dimension
        image_name = f"{img_name}_{dim_name}"

        # save to the wandb folder
        os.makedirs("wandb", exist_ok=True)
        save_path = f"wandb/{image_name}.png"
        cv2.imwrite(save_path, (slices * 255).astype(np.uint8))

        # Convert the figure to a WandB image object
        wandb_img = wandb.Image(slices, caption=image_name)
        # wandb_images.append(wandb_img)
        wandb_images[image_name] = wandb_img


        # # Upload the image to WandB
        # wandb.log({image_name: wandb_img})

    return wandb_images