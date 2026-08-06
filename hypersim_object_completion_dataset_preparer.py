import os
import glob
import random
import numpy as np
import h5py
from scipy.ndimage import binary_dilation, minimum_filter
import matplotlib.pyplot as plt


MAX_DEPTH_METERS = 20.0 # for indoor scenes we consider this to be the farthest point
MAX_OCCLUSION = 0.60 # an object should not be occluded more than this
MIN_OBJ_SIZE = 0.001 # ratio to object vs whole image in pixels
EPSILON_DEPTH = 0.001 # tolerance of one millimeter in depth
DEPTH_BOUNDARY_THRESHOLD = 0.95 # amount of boundary that should be in front


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


###################################################################


def list_rgb_depth_instance_in_hypersim(
            dataset_root: str
        ) -> list:
    print("Listing rgb-depth-instance...")
    grouped_filepaths = []

    # Get all scene directories (e.g., ai_001_001, ai_002_001)
    scene_dirs_pattern = os.path.join(dataset_root, "ai_*")
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
                    
    # grouped_filepaths = random.sample(grouped_filepaths, DATA_DOWN_SAMPLE_CNT)

    print(f"Total sample count {len(grouped_filepaths)}")
    return grouped_filepaths


def read_rgb_img_in_hypersim(
            rgb_path: str,
            exposure_value = 0.0
        ) -> np.ndarray:
    with h5py.File(rgb_path, 'r') as f:
        rgb_hdr = np.array(f["dataset"], dtype=np.float32)

    print(f"rgb NAN count: {np.isnan(rgb_hdr).sum()}")
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


def read_depth_map_in_hypersim(
            depth_map_path: str
        ) -> np.ndarray:
    with h5py.File(depth_map_path, 'r') as f:
        depth_map = np.array(f["dataset"], dtype=np.float32)

    # print(np.isnan(depth).sum())
    depth_map = np.nan_to_num( # missing/infinity depth is handled
        depth_map,
        nan=MAX_DEPTH_METERS,
        posinf=MAX_DEPTH_METERS,
        neginf=0.0
    )
    # bound outliers: lower than 0.0 are elevated to 0.0, and values exceeding
    # MAX_DEPTH_METERS are truncated to that maximum
    depth_map = np.clip(depth_map, a_min=0.0, a_max=MAX_DEPTH_METERS)
    
    # print(f"depth: {depth_map.shape}, min: {depth_map.min()}, max: {depth_map.max()}")
    return depth_map


def read_instance_mask_in_hypersim(
            instance_mask_path: str
        ) -> np.ndarray:
    with h5py.File(instance_mask_path, 'r') as f:
        instance_mask = np.array(f["dataset"], dtype=np.int32)

    # print(f"instance: {instance_mask.shape}, min: {instance_mask.min()}, max: {instance_mask.max()}")
    return instance_mask


###################################################################


def get_unoccluded_instance_mask(
            depth_map: np.ndarray,
            instance_mask: np.ndarray,
            bg_id = -1,
            small_obj_threshold = MIN_OBJ_SIZE,
            epsilon_depth = EPSILON_DEPTH,
            depth_boundary_threshold = DEPTH_BOUNDARY_THRESHOLD
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

        # if the object is beyond max depth
        if obj_depths.min() >= MAX_DEPTH_METERS:
            continue
        
        # Expand the object's depth outward by 1 pixel
        expanded_obj_depths = minimum_filter(obj_depths, footprint=structure)
        
        # Extract depths for object and its surrounding
        surround_depth = depth_map[surrounding_mask]
        adjacent_obj_depth = expanded_obj_depths[surrounding_mask]
        
        # ALL surrounding pixels must be further or equal to the adjacent object pixel
        valid_surrounding_pixels = surround_depth + epsilon_depth >= adjacent_obj_depth
        if np.mean(valid_surrounding_pixels) > depth_boundary_threshold:
            unoccluded_mask[obj_binary] = obj_id

    print(f"unoccluded instance: {np.unique(unoccluded_mask)}")
    return unoccluded_mask


def extract_rgb(
            rgb_image: np.ndarray,
            instance_mask: np.ndarray,
            bg_id = 0
        ) -> np.ndarray:
    instance_image = np.zeros_like(rgb_image)
    foreground_mask = (instance_mask != bg_id)
    instance_image[foreground_mask] = rgb_image[foreground_mask]
    
    return instance_image


def crop_out_mask(
            instance_mask: np.ndarray
        ) -> np.ndarray:
    y_mask, x_mask = np.where(instance_mask)
    y1, y2 = y_mask.min(), y_mask.max() + 1
    x1, x2 = x_mask.min(), x_mask.max() + 1

    instance_mask_crop = instance_mask[y1:y2, x1:x2]
    
    return instance_mask_crop


def shift_mask_randomly(
            occluder_crop: np.ndarray,
            rgb_target: np.ndarray,
        ) -> np.ndarray:
    y_t, x_t, z_t = np.where(rgb_target > 0)

    # Sometimes the object rgb is black, indistinguishable from the background
    if y_t.size == 0 or x_t.size == 0:
        print(f"! rgb_target: ({y_t}, {x_t})") # debug
        plot_np_arr([rgb_target, occluder_crop], "sd1_5_completion_empty_target_debug") # debug
        return None

    y1_t, y2_t = y_t.min(), y_t.max() + 1
    x1_t, x2_t = x_t.min(), x_t.max() + 1

    h_t, w_t, c_t = rgb_target.shape
    h_o, w_o = occluder_crop.shape

    min_overlap = 0.25
    
    min_y = max(0,                                    y1_t - h_o + ((y2_t - y1_t) * min_overlap))
    min_x = max(0,                                    x1_t - w_o + ((x2_t - x1_t) * min_overlap))
    max_y = min(y2_t - ((y2_t - y1_t) * min_overlap), h_t - h_o)
    max_x = min(x2_t - ((x2_t - x1_t) * min_overlap), w_t - w_o)
        
    # Sometimes the occluder may take the entire height or width
    if min_y == max_y:
        rand_y = min_y
    else:
        rand_y = np.random.randint(min_y, max_y)

    if min_x == max_x:
        rand_x = min_x
    else:
        rand_x = np.random.randint(min_x, max_x)
    
    shifted_occluder = np.zeros((h_t, w_t), dtype=bool)
    shifted_occluder[rand_y : rand_y + h_o, rand_x : rand_x + w_o] = occluder_crop

    return shifted_occluder


def generate_random_unoccluded_occluded_pair(
            rgb_img: np.ndarray,
            unoccluded_instance_mask: np.ndarray,
        ) -> list:
    instance_ids = np.unique(unoccluded_instance_mask)
    target_id_index = random.randint(1, len(instance_ids) - 1)
    occluder_id_index = random.randint(1, len(instance_ids) - 1)
    
    target_mask = (unoccluded_instance_mask == instance_ids[target_id_index])
    occluder_mask = (unoccluded_instance_mask == instance_ids[occluder_id_index])

    rgb_target = extract_rgb(rgb_img, target_mask)
    occluder_mask_cropped = crop_out_mask(occluder_mask)


    shifted_occluder_mask = shift_mask_randomly(occluder_mask_cropped, rgb_target)
    if shifted_occluder_mask is None:
        return None, None
    
    rgb_occluded = extract_rgb(rgb_target, np.logical_not(shifted_occluder_mask))

    # test
    plot_np_arr([rgb_img, unoccluded_instance_mask, rgb_target, shifted_occluder_mask, rgb_occluded], "hypersim_data")

    return rgb_target, rgb_occluded


# mostly returns the data_index, if the generated data is not "good" then returns a random data
def generate_one_random_item(
            grouped_filepaths: list,
            data_index: int,
        ) -> list:
    occlusion_ratio = 1.0
    unoccluded_rgb = None
    occluded_rgb = None
    while occlusion_ratio > MAX_OCCLUSION:
        exposure_value = random.uniform(-1.0, 1.0)
        print(f"## Data index {data_index}")

        rgb_img = read_rgb_img_in_hypersim(grouped_filepaths[data_index][0], exposure_value)
        depth_map = read_depth_map_in_hypersim(grouped_filepaths[data_index][1])
        instance_mask = read_instance_mask_in_hypersim(grouped_filepaths[data_index][2])

        # setting a random index for next loop, as the data generation scheme
        # of hypersim sometimes produces poor samples
        data_index = random.randint(0, len(grouped_filepaths) - 1)

        unoccluded_instance_mask = get_unoccluded_instance_mask(depth_map, instance_mask)
        if len(np.unique(unoccluded_instance_mask)) < 2:
            continue
        
        unoccluded_rgb, occluded_rgb = generate_random_unoccluded_occluded_pair(rgb_img, unoccluded_instance_mask)
        if unoccluded_rgb is None or occluded_rgb is None:
            continue
        
        occlusion_ratio = 1 - np.count_nonzero(occluded_rgb) / np.count_nonzero(unoccluded_rgb)
        print(f"occlusion_ratio: {occlusion_ratio}")
    
    return unoccluded_rgb, occluded_rgb
    

if __name__ == "__main__":
    dataset_root = "/workspace/minhas/dataset/hypersim/unzips/"
    grouped_filepaths = list_rgb_depth_instance_in_hypersim(dataset_root)
    unoccluded_rgb, occluded_rgb = generate_one_random_item(grouped_filepaths, 30_000)
