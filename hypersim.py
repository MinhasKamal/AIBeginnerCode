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


####--**--####


import os
import glob
import random
import h5py


HYPERSIM_DATASET_ROOT = "/workspace/minhas/dataset/hypersim/unzips/"
HYPERSIM_DATA_DOWN_SAMPLE_COUNT = -1 ## TEST ONLY!!!
def list_rgb_depth_instance_in_hypersim(
        ) -> list:
    print("# Listing rgb-depth-instance...")
    grouped_filepaths = []

    # Get all scene directories (e.g., ai_001_001, ai_002_001)
    scene_dirs_pattern = os.path.join(HYPERSIM_DATASET_ROOT, "ai_*")
    scene_dirs = glob.glob(scene_dirs_pattern)
    
    for scene_dir in sorted(scene_dirs):
        scene_dir = os.path.join(scene_dir, "images")
        if not os.path.exists(scene_dir):
            print(f"{scene_dir} does not exist!")
            continue
            
        # Find all final preview camera directories in this scene
        cam_preview_dirs_pattern = os.path.join(scene_dir, "scene_cam_*_final_hdf5")
        cam_preview_dirs = glob.glob(cam_preview_dirs_pattern)

        for rgb_cam_dir in cam_preview_dirs:
            # Extract the camera identifier (e.g., 'cam_00') to find its geometry match
            # depth_cam_dir = rgb_cam_dir[:-len("final_preview")] + "geometry_preview"
            depth_cam_dir = rgb_cam_dir[:-len("final_hdf5")] + "geometry_hdf5"
            
            if not os.path.exists(depth_cam_dir):
                print(f"{depth_cam_dir} does not exist!!")
                continue
                
            # Find all RGB frames inside this camera folder
            rgb_files_pattern = os.path.join(rgb_cam_dir, "frame.*.color.hdf5")
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

                instance_path = os.path.join(depth_cam_dir, f"frame.{frame_idx}.semantic_instance.hdf5")
                if not os.path.exists(instance_path):
                    print(f"{instance_path} does not exist!!!")
                    continue
                    
                grouped_filepaths.append((rgb_path, depth_path, instance_path))
                # break ## TEST ONLY!!!

    if HYPERSIM_DATA_DOWN_SAMPLE_COUNT > 0: ## TEST ONLY!!!
        grouped_filepaths = random.sample(grouped_filepaths, HYPERSIM_DATA_DOWN_SAMPLE_COUNT)

    print(f"Total sample count: {len(grouped_filepaths)}")
    return grouped_filepaths


def read_rgb_in_hypersim(
            rgb_path: str
        ) -> np.ndarray:
    with h5py.File(rgb_path, 'r') as f:
        rgb_hdr = np.array(f["dataset"], dtype=np.float32)

    # print(f"rgb NAN count: {np.isnan(rgb_hdr).sum()}")
    rgb_hdr = np.nan_to_num( # missing/infinity instance is handled
        rgb_hdr, 
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # randomly set an exposure value
    exposure_value = random.uniform(-1.0, 1.0)

    # brightness adjustment: I = I * 2^(EV)
    # +1 exposure value doubles the light, -1 exposure value halves the light
    rgb_hdr = rgb_hdr * (2.0 ** exposure_value)

    # Standard sRGB Gamma Correction: Linear (High Dynamic Range) -> Low Dynamic Range
    # The function (Opto-Electronic Transfer Function) is the official linear-to-sRGB
    # (standard RGB) conversion standardized by Hewlett-Packard and Microsoft.
    # x <= 0.0031308 ? 12.92 * x : 1.055 * (x ** (1 / 2.4)) - 0.055
    rgb_ldr = np.where(
        rgb_hdr <= 0.0031308,
        12.92 * rgb_hdr,
        1.055 * np.power(rgb_hdr, 1.0 / 2.4) - 0.055
    )
    rgb_8bit = np.clip(rgb_ldr * 255.0, 0, 255).astype(np.uint8)

    rgb_8bit = np.transpose(rgb_8bit, (2, 0, 1)) # (H, W, Ch) -> (Ch, H, W)

    # print(f"rgb: {rgb_8bit.shape}, min: {rgb_8bit.min()}, max: {rgb_8bit.max()}")
    return rgb_8bit


HYPERSIM_MAX_DEPTH_METER = 1_000.0
def read_depth_in_hypersim(
            depth_map_path: str
        ) -> np.ndarray:
    with h5py.File(depth_map_path, 'r') as f:
        depth_map_meter = np.array(f["dataset"], dtype=np.float32)

    # print(np.isnan(depth).sum())
    depth_map_meter = np.nan_to_num( # missing/infinity depth is handled
        depth_map_meter,
        nan=HYPERSIM_MAX_DEPTH_METER,
        posinf=HYPERSIM_MAX_DEPTH_METER,
        neginf=0.0
    )
    # bound outliers: lower than 0.0 are elevated to 0.0, and values exceeding
    # HYPERSIM_MAX_DEPTH_METER are truncated to that maximum
    depth_map_meter = np.clip(depth_map_meter, a_min=0.0, a_max=HYPERSIM_MAX_DEPTH_METER)
    
    # print(f"depth: {depth_map_meter.shape}, min: {depth_map_meter.min()}, max: {depth_map_meter.max()}")
    return depth_map_meter


def read_instance_mask_in_hypersim(
            instance_mask_path: str
        ) -> np.ndarray:
    with h5py.File(instance_mask_path, 'r') as f:
        instance_mask = np.array(f["dataset"], dtype=np.int32)

    # print(f"instance: {instance_mask.shape}, min: {instance_mask.min()}, max: {instance_mask.max()}")
    return instance_mask


####--**--####


if __name__ == "__main__":
    dataset_root = "/workspace/minhas/dataset/hypersim/unzips/"
    paired_filepaths = list_rgb_depth_pairs_in_hypersim(dataset_root)
    rgb_path, depth_path = paired_filepaths[0]
    transform_image_pair(rgb_path, depth_path)
