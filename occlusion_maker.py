import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt


def read_rgb_in_hypersim(
            rgb_path: str,
            exposure_value = 0.0
        ) -> np.ndarray:
    with h5py.File(rgb_path, 'r') as f:
        rgb_hdr = np.array(f["dataset"], dtype=np.float32)

    print(f"NAN count: {np.isnan(rgb_hdr).sum()}")
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


def read_instance_in_hypersim(
            instance_path: str
        ) -> np.ndarray:
    with h5py.File(instance_path, 'r') as f:
        instance = np.array(f["dataset"], dtype=np.int32)

    print(f"instance: {instance.shape}, min: {instance.min()}, max: {instance.max()}")
    return instance


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
    y_t, x_t, z_t = np.where(rgb_target)
    y1_t, y2_t = y_t.min(), y_t.max() + 1
    x1_t, x2_t = x_t.min(), x_t.max() + 1

    h_t, w_t, c_t = rgb_target.shape
    h_o, w_o = occluder_crop.shape

    min_overlap = 0.25
    
    min_y = max(0,                                    y1_t - h_o + ((y2_t - y1_t) * min_overlap))
    min_x = max(0,                                    x1_t - w_o + ((x2_t - x1_t) * min_overlap))
    max_y = min(y2_t - ((y2_t - y1_t) * min_overlap), h_t - h_o)
    max_x = min(x2_t - ((x2_t - x1_t) * min_overlap), w_t - w_o)
    
    rand_y = np.random.randint(min_y, max_y)
    rand_x = np.random.randint(min_x, max_x)

    # shifted_occluder = np.zeros_like(rgb_target, dtype=bool)
    # shifted_occluder[rand_y : rand_y + h_o, rand_x : rand_x + w_o] = np.stack(
    #     (occluder_crop, occluder_crop, occluder_crop), axis=-1)
    
    shifted_occluder = np.zeros((h_t, w_t), dtype=bool)
    shifted_occluder[rand_y : rand_y + h_o, rand_x : rand_x + w_o] = occluder_crop

    return shifted_occluder


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


if __name__ == "__main__":
    rgb_path = "/workspace/minhas/dataset/hypersim/unzips/" \
            "ai_001_003/images/scene_cam_00_final_hdf5/frame.0000.color.hdf5"
    instance_path = "/workspace/minhas/dataset/hypersim/unzips/" \
            "ai_001_003/images/scene_cam_00_geometry_hdf5/frame.0000.semantic_instance.hdf5"

    exposure_value = 0.0 # 1.0, -1.0
    rgb_img = read_rgb_in_hypersim(rgb_path, exposure_value)
    instance_mask = read_instance_in_hypersim(instance_path)

    instance_ids = np.unique(instance_mask)
    instance_mask_target = (instance_mask == instance_ids[2])
    instance_mask_occluder = (instance_mask == instance_ids[3])

    rgb_target = extract_rgb(rgb_img, instance_mask_target)
    instance_mask_occluder_crop = crop_out_mask(instance_mask_occluder)

    shifted_occluder = shift_mask_randomly(instance_mask_occluder_crop, rgb_target)
    
    masked_obj = extract_rgb(rgb_target, np.logical_not(shifted_occluder))
    plot_np_arr([rgb_target, instance_mask_occluder_crop, shifted_occluder, masked_obj], "occluded_mask")
