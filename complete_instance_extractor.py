import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, minimum_filter


MAX_DEPTH_METERS = 20.0


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

    # print(np.isnan(instance).sum())
    # instance = np.nan_to_num( # missing/infinity instance is handled
    #     instance, 
    #     nan=-1.0,
    #     posinf=-1.0,
    #     neginf=-1.0
    # )

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
            np_arr: np.ndarray,
            file_name: str
        ):

    plt.imshow(np_arr, cmap="gray")
    plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.clf()
    
    return

def plot_img(
            depth_img: Image, 
            instance_img: Image, 
            instance_img_unoccluded: Image, 
            file_name: str
        ):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    # axes[0].imshow(depth_img)
    # axes[1].imshow(instance_img)
    axes[0].imshow(depth_img, cmap='gray')
    axes[1].imshow(instance_img, cmap='gray')
    axes[2].imshow(instance_img_unoccluded, cmap='gray')
    # plt.tight_layout()
    plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    plt.clf()
    
    return


def get_unoccluded_instances(
            depth_map: np.ndarray,
            instance_mask: np.ndarray,
            bg_id: int = -1
        ) -> np.ndarray:
    """
    Strictly filters unoccluded instances for perfect synthetic datasets (like Hypersim).
    
    Args:
        depth_map (np.ndarray): 2D array of exact depth values (smaller = closer).
        instance_mask (np.ndarray): 2D array of instance labels.
        bg_id (int): ID representing the background/invalid class.
        
    Returns:
        np.ndarray: A new instance mask containing ONLY unoccluded objects.
    """
    unoccluded_mask = np.full_like(instance_mask, bg_id)
    
    object_ids = np.unique(instance_mask)
    print(f"all instances: {object_ids}")
    object_ids = object_ids[object_ids != bg_id]
    
    # 3x3 footprint for exactly 1-pixel boundary math
    structure = np.ones((3, 3), dtype=bool)
    
    for obj_id in object_ids:
        obj_binary = (instance_mask == obj_id)
        plot_np_arr(obj_binary, "co_test")
        break
        
        # # 1. Get the exact 1-pixel surrounding halo
        # dilated_mask = binary_dilation(obj_binary, structure=structure)
        # surrounding_mask = dilated_mask ^ obj_binary
        
        # # If the object fills the frame or has no surroundings, it's unoccluded
        # if not np.any(surrounding_mask):
        #     unoccluded_mask[obj_binary] = obj_id
        #     continue
            
        # # 2. Isolate the object's depths (set everything else to infinity)
        # obj_depths = np.full_like(depth_map, np.inf)
        # obj_depths[obj_binary] = depth_map[obj_binary]
        
        # # 3. Expand the object's depth outward by 1 pixel.
        # # This assigns each pixel in the halo the exact depth of the closest object edge.
        # expanded_obj_depths = minimum_filter(obj_depths, footprint=structure)
        
        # # 4. Extract depths strictly at the halo
        # surround_depth = depth_map[surrounding_mask]
        # adjacent_obj_depth = expanded_obj_depths[surrounding_mask]
        
        # # 5. Perfect Data Check: ALL surrounding pixels must be further or equal 
        # # to the adjacent object pixel. (No noise thresholds needed).
        # if np.all(surround_depth >= adjacent_obj_depth):
        #     unoccluded_mask[obj_binary] = obj_id

    print(f"unoccluded instance: {np.unique(unoccluded_mask)}")
    return unoccluded_mask
    

if __name__ == "__main__":
    depth_path = "/workspace/minhas/dataset/hypersim/unzips/" \
            "ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5"
    instance_path = "/workspace/minhas/dataset/hypersim/unzips/" \
            "ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.semantic_instance.hdf5"

    depth_map = read_depth_in_hypersim(depth_path)
    instance_mask = read_instance_in_hypersim(instance_path)

    depth_img = np_arr_to_img(depth_map)
    instance_img = np_arr_to_img(instance_mask)

    instance_mask_unoccluded = get_unoccluded_instances(depth_map, instance_mask)
    instance_img_unoccluded = np_arr_to_img(instance_mask_unoccluded)

    plot_img(depth_img, instance_img, instance_img_unoccluded, "depth_instance_preview")
    