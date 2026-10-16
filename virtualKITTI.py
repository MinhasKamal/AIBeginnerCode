

####--**--####


import matplotlib.pyplot as plt


def plot_np_arr_list(
            np_arr_list: list,
            file_name: str
        ):
    fig, axes = plt.subplots(1, len(np_arr_list), figsize=(6 * len(np_arr_list), 10))

    if len(np_arr_list) > 1:
        for index, np_arr in enumerate(np_arr_list):            
            axes[index].imshow(np_arr)
            # axes[index].imshow(np_arr, cmap="gray")
    else :
        axes.imshow(np_arr_list[0])
        # axes.imshow(np_arr_list[0], cmap="gray")
    
    plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.clf()
    
    return


####--**--####


import os
import glob
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


VKITTI_DATASET_ROOT = "/workspace/minhas/dataset/virtual_kitti/"
VKITTI_DATA_DOWN_SAMPLE_COUNT = -1 ## TEST ONLY!!!
def list_rgb_depth_in_vkitti(
        ) -> list:
    
    print("# Listing rgb-depth in Virtual KITTI...")
    grouped_filepaths = []
    
    # Get all scene directories (e.g., Scene01, Scene02, Scene06, etc.)
    scene_dirs_pattern = os.path.join(VKITTI_DATASET_ROOT, "Scene*")
    scene_dirs = glob.glob(scene_dirs_pattern)
    
    for scene_dir in sorted(scene_dirs):
        if not os.path.isdir(scene_dir):
            continue
            
        # vKITTI2 scenes have variation subfolders (e.g., 'clone', 'fog', 'morning', '15-deg-left')
        variation_dirs = glob.glob(os.path.join(scene_dir, "*"))
        
        for variation_dir in sorted(variation_dirs):
            if not os.path.isdir(variation_dir):
                continue
                
            # Paths to rgb and depth frame directories
            rgb_base_dir = os.path.join(variation_dir, "frames", "rgb")
            depth_base_dir = os.path.join(variation_dir, "frames", "depth")
            
            if not os.path.exists(rgb_base_dir) or not os.path.exists(depth_base_dir):
                continue
                
            # Iterate through cameras (e.g., Camera_0, Camera_1)
            camera_dirs = glob.glob(os.path.join(rgb_base_dir, "Camera_*"))
            
            for rgb_cam_dir in sorted(camera_dirs):
                cam_name = os.path.basename(rgb_cam_dir) # e.g., 'Camera_0'
                depth_cam_dir = os.path.join(depth_base_dir, cam_name)
                
                if not os.path.exists(depth_cam_dir):
                    print(f"{depth_cam_dir} does not exist!!")
                    continue
                    
                # Find all RGB frames inside this camera folder (vKITTI2 RGBs are usually .jpg)
                rgb_files_pattern = os.path.join(rgb_cam_dir, "rgb_*.jpg")
                rgb_files = glob.glob(rgb_files_pattern)
                
                # Fallback in case your specific download uses .png for RGB
                # if not rgb_files:
                #     rgb_files_pattern = os.path.join(rgb_cam_dir, "rgb_*.png")
                #     rgb_files = glob.glob(rgb_files_pattern)
                
                for rgb_path in rgb_files:
                    # Extract the exact frame number (e.g., '00000') from the filename 'rgb_00000.jpg'
                    filename = os.path.basename(rgb_path)
                    frame_idx = filename.split("_")[1].split(".")[0]
                    
                    # Construct the expected matching depth map path (vKITTI2 depths are 16-bit .png)
                    depth_path = os.path.join(depth_cam_dir, f"depth_{frame_idx}.png")
                    
                    # Verify that the depth map physically exists before adding the pair
                    if not os.path.exists(depth_path):
                        print(f"{depth_path} does not exist!!!")
                        continue
                        
                    grouped_filepaths.append((rgb_path, depth_path))
                    # break ## TEST ONLY!!!

    if VKITTI_DATA_DOWN_SAMPLE_COUNT > 0: ## TEST ONLY!!!
        grouped_filepaths = random.sample(grouped_filepaths, VKITTI_DATA_DOWN_SAMPLE_COUNT)
        
    print(f"Total sample count: {len(grouped_filepaths)}")
    return grouped_filepaths


def read_rgb_in_vkitti(
        rgb_path: str
    ) -> np.ndarray:
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_8bit = np.array(rgb_img, dtype=np.uint8)
    rgb_8bit = np.transpose(rgb_8bit, (2, 0, 1)) # (H, W, Ch) -> (Ch, H, W)
    return rgb_8bit


VKITTI_MAX_DEPTH_METERS: float = 1_000.0
def read_depth_in_vkitti(
            depth_path: str
        ) -> np.ndarray:

    depth_map_centimeter = np.array(Image.open(depth_path), dtype=np.float32)
    depth_map_meter = depth_map_centimeter / 100.0
    depth_map_meter = np.expand_dims(depth_map_meter, axis=0) # [H, W] -> [1, H, W]

    # Map 0.0 (sky or invalid pixels) to max depth
    depth_map_meter[depth_map_meter == 0.0] = VKITTI_MAX_DEPTH_METERS

    depth_map_meter = np.nan_to_num( # missing or infinity depth is handled
        depth_map_meter,
        nan=VKITTI_MAX_DEPTH_METERS,
        posinf=VKITTI_MAX_DEPTH_METERS,
        neginf=0.0
    )

    # bound outliers: lower than 0.0 are elevated to 0.0, and values exceeding
    # MAX_DEPTH_METERS are truncated to that maximum
    depth_map_meter = np.clip(depth_map_meter, 0.0, VKITTI_MAX_DEPTH_METERS)

    # print(depth_map_meter.shape)
    return depth_map_meter


####--**--####


if __name__ == "__main__":
    
    rgb_depth_filepaths = list_rgb_depth_in_vkitti()
    print(rgb_depth_filepaths[0])
    
    rgb_8bit = read_rgb_in_vkitti(rgb_depth_filepaths[0][0])
    depth_map_meter = read_depth_in_vkitti(rgb_depth_filepaths[0][1])
    # Matplotlib expects (h, w, ch) or (h, w), but our image is shaped as (ch, h, w)
    plot_np_arr_list([rgb_8bit.transpose(1, 2, 0), np.squeeze(depth_map_meter)], "virtualKITTI_test")
    
    