import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import h5py


MAX_DEPTH_METERS = 20.0
DATA_DOWN_SAMPLE_CNT = 10


def list_rgb_depth_pairs_in_hypersim(
            dataset_root: str
        ) -> list:
    print("Listing rgb-depth pairs...")
    paired_filepaths = []

    # Get all scene directories (e.g., ai_001_001, ai_002_001)
    scene_dirs_pattern = os.path.join(dataset_root, "ai_*")
    scene_dirs = glob.glob(scene_dirs_pattern)
    
    for scene_dir in sorted(scene_dirs):
        scene_dir = os.path.join(scene_dir, "images")
        if not os.path.exists(scene_dir):
            print(f"{scene_dir} does not exist!")
            continue
            
        # Find all final preview camera directories in this scene
        cam_preview_dirs_pattern = os.path.join(scene_dir, "scene_cam_*_final_preview")
        cam_preview_dirs = glob.glob(cam_preview_dirs_pattern)

        for rgb_cam_dir in cam_preview_dirs:
            # Extract the camera identifier (e.g., 'cam_00') to find its geometry match
            # depth_cam_dir = rgb_cam_dir[:-len("final_preview")] + "geometry_preview"
            depth_cam_dir = rgb_cam_dir[:-len("final_preview")] + "geometry_hdf5"
            
            if not os.path.exists(depth_cam_dir):
                print(f"{depth_cam_dir} does not exist!!")
                continue
                
            # Find all RGB frames inside this camera folder
            rgb_files_pattern = os.path.join(rgb_cam_dir, "frame.*.color.jpg")
            rgb_files = glob.glob(rgb_files_pattern)
            
            for rgb_path in rgb_files:
                # Extract the exact frame number (e.g., '0000', '0001') from the filename
                filename = os.path.basename(rgb_path)
                frame_idx = filename.split(".")[1]
                
                # Construct the expected matching depth map path
                # depth_path = os.path.join(depth_cam_dir, f"frame.{frame_idx}.depth_meters.png")
                depth_path = os.path.join(depth_cam_dir, f"frame.{frame_idx}.depth_meters.hdf5")
                
                # Verify that the depth map physically exists before adding the pair
                if not os.path.exists(depth_path):
                    print(f"{depth_path} does not exist!!!")
                    continue
                    
                paired_filepaths.append((rgb_path, depth_path))
                break ########################################################################################
                    
    paired_filepaths = random.sample(paired_filepaths, DATA_DOWN_SAMPLE_CNT)
    print(f"Total rgb-depth frame pairs {len(paired_filepaths)}")
    # print(paired_filepaths)
    return paired_filepaths


def tensor_to_img(tensor):
    tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    
    # Convert tensor to a PIL Image
    tensor = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    
    img = Image.fromarray(tensor)
    return img


def plot_img(rgb_img, depth_img, file_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[1].imshow(depth_img)
    # plt.tight_layout()
    plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    plt.clf()
    
    return


def read_depth_in_hypersim(
            depth_path: str
        ) -> torch.Tensor:
    # depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    # depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    with h5py.File(depth_path, 'r') as f:
        depth_raw = f['dataset'][:]
    
    depth_tensor = torch.from_numpy(depth_raw.astype(np.float32))
    depth_tensor = depth_tensor.unsqueeze(0) # [H, W] -> [1, H, W]
    depth_tensor = torch.nan_to_num( # missing/infinity depth is handled
        depth_tensor, 
        nan=MAX_DEPTH_METERS, 
        posinf=MAX_DEPTH_METERS, 
        neginf=0.0
    )
    # bound outliers: lower than 0.0 are elevated to 0.0, and values exceeding
    # MAX_DEPTH_METERS are truncated to that maximum
    depth_tensor = torch.clamp(depth_tensor, 0.0, MAX_DEPTH_METERS)
    depth_tensor = (depth_tensor / MAX_DEPTH_METERS) * 2.0 - 1.0 # Normalization to [-1.0, 1.0]
    depth_tensor = depth_tensor.repeat(3, 1, 1) # 3 channels for the VAE: [1, H, W] -> [3, H, W]
    # print(depth_tensor.shape)
    return depth_tensor

def transform_image_pair(rgb_path, depth_path):
    image_size = 512
    crop_len = image_size // 10
    noise_std = 0.02

    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_tensor = TF.to_tensor(rgb_img) # (H, W, Ch) -> (Ch, H, W) & scale to [0.0, 1.0]
    rgb_tensor = (rgb_tensor * 2 - 1.0) # [0.0, 1.0] -> [-1.0, 1.0]
    # depth_img = Image.open(depth_path).convert("RGB")
    depth_tensor = read_depth_in_hypersim(depth_path)
    plot_img(tensor_to_img(rgb_tensor), tensor_to_img(depth_tensor), "hypersim_01_raw")

    target_size = (image_size + crop_len, image_size + crop_len)
    # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
    # # So we unsqueeze(0) to fake a batch of 1, interpolate, and squeeze(0) to remove it
    rgb_tensor = F.interpolate(
        rgb_tensor.unsqueeze(0), 
        size=target_size,
        mode='nearest',
        # mode='bilinear', 
        # align_corners=False
    ).squeeze(0)
    depth_tensor = F.interpolate(
        depth_tensor.unsqueeze(0), 
        size=target_size,
        mode='nearest',
        # mode='bilinear', 
        # align_corners=False
    ).squeeze(0)
    plot_img(tensor_to_img(rgb_tensor), tensor_to_img(depth_tensor), "hypersim_02_resized")

    # Random crop out
    top = random.randint(0, crop_len)
    left = random.randint(0, crop_len)
    rgb_tensor = TF.crop(rgb_tensor, top, left, image_size, image_size)
    depth_tensor = TF.crop(depth_tensor, top, left, image_size, image_size)
    plot_img(tensor_to_img(rgb_tensor), tensor_to_img(depth_tensor), "hypersim_03_cropped")

    # Apply random flip
    if random.random() > 0.5:
        rgb_tensor = TF.hflip(rgb_tensor)
        depth_tensor = TF.hflip(depth_tensor)
    plot_img(tensor_to_img(rgb_tensor), tensor_to_img(depth_tensor), "hypersim_04_flipped")

    # Add subtle Gaussian noise to RGB sampled from N(0, noise_std)
    noise = torch.randn_like(rgb_tensor) * noise_std
    rgb_tensor = rgb_tensor + noise
    # Adding noise to a pixel value of 1.0 can push it to 1.02
    rgb_tensor = torch.clamp(rgb_tensor, -1.0, 1.0)
    plot_img(tensor_to_img(rgb_tensor), tensor_to_img(depth_tensor), "hypersim_05_noised")

    return


if __name__ == "__main__":
    dataset_root = "/workspace/minhas/dataset/hypersim/unzips/"
    paired_filepaths = list_rgb_depth_pairs_in_hypersim(dataset_root)
    rgb_path, depth_path = paired_filepaths[0]
    transform_image_pair(rgb_path, depth_path)
