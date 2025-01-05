import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

class ImageVisualizer:

    @staticmethod
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
            print(slices.shape)
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

            # Convert the figure to a WandB image object
            wandb_img = wandb.Image(slices, caption=image_name)
            # wandb_images.append(wandb_img)
            wandb_images[image_name] = wandb_img


            # # Upload the image to WandB
            # wandb.log({image_name: wandb_img})

        return wandb_images

# # Example usage
# if __name__ == "__main__":
#     # Initialize WandB
#     wandb.init(project="3d_image_visualization", name="image_visualization_test")
    
#     # Create some dummy 3D images (10 random 3D tensors of shape (64, 64, 64))
#     img_list = [torch.rand(64, 128, 32) for _ in range(3)]
    
#     # Call the function
#     vis_3d_images = ImageVisualizer.vis_3d_img_list(img_list, img_name="test_image")
    
#     # Finalize WandB
#     wandb.finish()



# # Example usage
# if __name__ == "__main__":
#     # Initialize WandB
#     wandb.init(project="3d_image_visualization", name="image_visualization_test_with_mask")

#     # Create a random 3D image (size: 64x64x64)
#     random_img = torch.rand(64, 64, 64)

#     # Create a random 3D binary mask image (size: 64x64x64)
#     mask_img = torch.randint(0, 2, (64, 64, 64), dtype=torch.float32)

#     # Apply the mask to the random image (set the masked areas to 0)
#     masked_img = random_img * mask_img

#     # Stack the images into a list for visualization
#     img_list = [random_img, mask_img, masked_img]

#     # Set the name for the images
#     img_name = "3d_image_with_mask"

#     # Call the visualization function
#     vis_3d_images = ImageVisualizer.vis_3d_img_list(img_list, img_name=img_name)

#     # Finalize WandB
#     wandb.finish()

# Example usage
if __name__ == "__main__":
    # Initialize WandB
    wandb.init(project="3d_image_visualization", name="NP_image_visualization_test_with_half_mask")

    # import nibabel as nib

    img_path = "/home/xufluo/ct-clip-vit/ct_rate/sub_sample_30_dataset/train_img_pre/train_1/train_1/train_1a/train_1_a_1.npz"

    img_arr = np.load(img_path)["arr_0"]

    # nomalize the image to 0-1
    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())

    # as tensor

    img_arr = torch.tensor(img_arr)

    print(img_arr.shape)

    # exit()

    # Create a random 3D image (size: 64x64x64)
    random_img = img_arr

    # Create a 3D binary mask image (size: 64x64x64)
    mask_img = torch.zeros(random_img.shape)

    D, H, W = random_img.shape

    # Set the first half of each dimension to 1 in the mask
    # mask_img[:32, :, :] = 1  # First half along dimension 0
    # mask_img[:, :32, :] = 1  # First half along dimension 1
    # mask_img[:, :, :32] = 1  # First half along dimension 2

    mask_img[D//2:, :, :] = 1  # First half along dimension 0
    mask_img[:, H//2:, :] = 1  # First half along dimension 1
    mask_img[:, :, W//2:] = 1  # First half along dimension 2

    # Apply the mask to the random image (set the masked areas to 0)
    masked_img = random_img * mask_img

    # Stack the images into a list for visualization
    img_list = [random_img, mask_img, masked_img]

    # Set the name for the images
    img_name = "3d_image_with_half_mask"

    # Call the visualization function
    vis_3d_images = ImageVisualizer.vis_3d_img_list(img_list, img_name=img_name)

    # Finalize WandB
    wandb.finish()
