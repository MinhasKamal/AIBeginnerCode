import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, minimum_filter


MAX_DEPTH_METERS = 20.0


def read_rgb_in_hypersim(
            rgb_path: str,
            exposure_value = 0.0
        ) -> np.ndarray:
    with h5py.File(rgb_path, 'r') as f:
        rgb_hdr = np.array(f["dataset"], dtype=np.float32)

    print(np.isnan(rgb_hdr).sum())
    rgb_hdr = np.nan_to_num( # missing/infinity instance is handled
        rgb_hdr, 
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

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

    print(f"rgb: {rgb_8bit.shape}, min: {rgb_8bit.min()}, max: {rgb_8bit.max()}")
    return rgb_8bit


def read_depth_in_hypersim(
            depth_path: str
        ) -> np.ndarray:
    with h5py.File(depth_path, 'r') as f:
        depth = np.array(f["dataset"], dtype=np.float32)

    # print(np.isnan(depth).sum())
    depth = np.nan_to_num( # missing/infinity depth is handled
        depth,
        nan=MAX_DEPTH_METERS,
        posinf=MAX_DEPTH_METERS,
        neginf=0.0
    )
    # bound outliers: lower than 0.0 are elevated to 0.0, and values exceeding
    # MAX_DEPTH_METERS are truncated to that maximum
    depth = np.clip(depth, a_min=0.0, a_max=MAX_DEPTH_METERS)
    
    print(f"depth: {depth.shape}, min: {depth.min()}, max: {depth.max()}")
    return depth


def read_instance_in_hypersim(
            instance_path: str
        ) -> np.ndarray:
    with h5py.File(instance_path, 'r') as f:
        instance = np.array(f["dataset"], dtype=np.int32)

    print(f"instance: {instance.shape}, min: {instance.min()}, max: {instance.max()}")
    return instance


def np_arr_to_img(
            np_arr: np.ndarray
        ) -> Image:
    eps = 1e-8  # Prevents division by zero if all values are identical
    normalized_np = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min() + eps)
    normalized_np = np.clip(normalized_np, a_min=0.0, a_max=1.0)
    
    # Convert tensor to a PIL Image
    normalized_np = (normalized_np * 255).astype(np.uint8)

    img = Image.fromarray(normalized_np)
    return img


def plot_np_arr(
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


def get_unoccluded_instance_mask(
            depth_map: np.ndarray,
            instance_mask: np.ndarray,
            bg_id = -1,
            small_obj_threshold = 0.001, # ratio to object vs whole image in pixel
            epsilon_depth = 0.001, # tolerance of one millimeter in depth
            threshold_boundary = 0.95 # amount of boundary that should be in front
        ) -> np.ndarray:
    unoccluded_mask = np.full_like(instance_mask, bg_id)
    
    object_ids = np.unique(instance_mask)
    print(f"all instances: {object_ids}")
    object_ids = object_ids[object_ids != bg_id]
    
    # 3x3 footprint for exactly 1-pixel boundary math
    structure = np.ones((3, 3), dtype=bool)
    
    for obj_id in object_ids:
        # Get each object
        obj_binary = (instance_mask == obj_id)

        # Get rid of small objects
        if np.sum(obj_binary) < obj_binary.size * small_obj_threshold:
            # print(f"{obj_id}: {np.sum(obj_binary)} / {obj_binary.size}")
            continue
        
        # Get the exact 1-pixel surrounding the object
        dilated_mask = binary_dilation(obj_binary, structure=structure)
        surrounding_mask = dilated_mask ^ obj_binary
        
        # If the object fills the frame or has no surroundings, don't include it
        if not np.any(surrounding_mask):
            continue
            
        # Isolate the object's depths (set everything else to infinity)
        obj_depths = np.full_like(depth_map, np.inf)
        obj_depths[obj_binary] = depth_map[obj_binary]
        # obj_depths[surrounding_mask] = depth_map[surrounding_mask]
        
        # Expand the object's depth outward by 1 pixel
        expanded_obj_depths = minimum_filter(obj_depths, footprint=structure)
        
        # Extract depths for object and its surrounding
        surround_depth = depth_map[surrounding_mask]
        adjacent_obj_depth = expanded_obj_depths[surrounding_mask]
        
        # ALL surrounding pixels must be further or equal to the adjacent object pixel
        valid_surrounding_pixels = surround_depth + epsilon_depth >= adjacent_obj_depth
        if np.mean(valid_surrounding_pixels) > threshold_boundary:
            unoccluded_mask[obj_binary] = obj_id

    print(f"unoccluded instance: {np.unique(unoccluded_mask)}")
    return unoccluded_mask


def extract_instance_rgb(
            rgb_image: np.ndarray,
            instance_mask: np.ndarray,
            bg_id = -1
        ) -> np.ndarray:
    instance_image = np.zeros_like(rgb_image)
    foreground_mask = (instance_mask != bg_id)
    instance_image[foreground_mask] = rgb_image[foreground_mask]
    
    return instance_image


if __name__ == "__main__":
    rgb_path = "/workspace/minhas/dataset/hypersim/unzips/" \
            "ai_001_003/images/scene_cam_00_final_hdf5/frame.0000.color.hdf5"
    depth_path = "/workspace/minhas/dataset/hypersim/unzips/" \
            "ai_001_003/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5"
    instance_path = "/workspace/minhas/dataset/hypersim/unzips/" \
            "ai_001_003/images/scene_cam_00_geometry_hdf5/frame.0000.semantic_instance.hdf5"

    exposure_value = 0.0 # 1.0, -1.0
    rgb_img = read_rgb_in_hypersim(rgb_path, exposure_value)
    depth_map = read_depth_in_hypersim(depth_path)
    instance_mask = read_instance_in_hypersim(instance_path)

    # depth_img = np_arr_to_img(depth_map)
    # instance_img = np_arr_to_img(instance_mask)

    unoccluded_instance_mask = get_unoccluded_instance_mask(depth_map, instance_mask)
    # unoccluded_instance_img = np_arr_to_img(unoccluded_instance_mask)

    instance_img = extract_instance_rgb(rgb_img, unoccluded_instance_mask)
    plot_np_arr([depth_map, instance_mask, unoccluded_instance_mask, instance_img], "depth_instance_rgb_preview")
