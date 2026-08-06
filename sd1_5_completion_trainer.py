import os
import glob
import random
import numpy as np
import h5py
from scipy.ndimage import binary_dilation, minimum_filter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import PNDMScheduler
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer
from transformers import CLIPTextModel
import torchvision.transforms.functional as TF
from tqdm import tqdm
import json


vae_path = "vae_sd1-5_AutoencoderKL"
unet_path = "unet_sd1-5_UNet2DConditionModel"
scheduler_path = "scheduler_sd1-5_PNDMScheduler"
tokenizer_path = "tokenizer_sd1-5_CLIPTokenizer"
text_encoder_path = "text_encoder_sd1-5_CLIPTextModel"

OUT_MODEL_PATH = "completion_sd1-5_0"

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-5

DATASET_ROOT = "/workspace/minhas/dataset/hypersim/unzips/"
SD1_5_IMAGE_SIZE = 512
DATA_LOADER_WORKERS = 2
DATA_DOWN_SAMPLE_CNT = 10_000 #-1

PLT_DATA_SKIP = 100

UNET_LAYER1_WEIGHT_DIST_FOR_UNOCCLUDED = 0.5
CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE = 0.0

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


#################################################################################################################


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

    if DATA_DOWN_SAMPLE_CNT > 0:
        grouped_filepaths = random.sample(grouped_filepaths, DATA_DOWN_SAMPLE_CNT)

    print(f"Total sample count {len(grouped_filepaths)}")
    return grouped_filepaths


def read_rgb_img_in_hypersim(
            rgb_path: str,
            exposure_value = 0.0
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

    # print(f"rgb: {rgb_8bit.shape}, min: {rgb_8bit.min()}, max: {rgb_8bit.max()}")
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


#################################################################################################################
    

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
    # print(f"all instances: {object_ids}")
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

    # print(f"unoccluded instance: {np.unique(unoccluded_mask)}")
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
        # print(f"! rgb_target: ({y_t}, {x_t})")
        # plot_np_arr([rgb_target, occluder_crop], "sd1_5_completion_empty_target_debug")
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
    # plot_np_arr([rgb_img, unoccluded_instance_mask, rgb_target, shifted_occluder_mask, rgb_occluded], "hypersim_data")

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
        # print(f"## Data index {data_index}")

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
        # print(f"occlusion_ratio: {occlusion_ratio}")
    
    return unoccluded_rgb, occluded_rgb


#################################################################################################################


class RGBCompletionDataset(Dataset):
    def __init__(
                self, 
                rgb_depth_instance_filepaths: list,
                image_size: int,
            ) -> None:
        self.rgb_depth_instance_filepaths = rgb_depth_instance_filepaths
        self.image_size = image_size
        return

    def __len__(
                self
            ) -> int:
        return len(self.rgb_depth_instance_filepaths)

    def __getitem__(
                self,
                idx: int
            ) -> tuple[torch.Tensor, torch.Tensor]:

        unoccluded_rgb, occluded_rgb = generate_one_random_item(self.rgb_depth_instance_filepaths, idx)

        unoccluded_tensor = torch.from_numpy(unoccluded_rgb).float()
        occluded_tensor = torch.from_numpy(occluded_rgb).float()

        # (H, W, Ch) -> (Ch, H, W)
        unoccluded_tensor = unoccluded_tensor.permute(2, 0, 1)
        occluded_tensor = occluded_tensor.permute(2, 0, 1)

        # scale to [-1.0, 1.0]
        unoccluded_tensor = ((unoccluded_tensor / 255.0) * 2.0) - 1.0
        occluded_tensor = ((occluded_tensor / 255.0) * 2.0) - 1.0

        target_size = (self.image_size, self.image_size)
        # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
        # # So we unsqueeze(0) to fake a batch of 1, interpolate, and squeeze(0) to remove it
        unoccluded_tensor = F.interpolate(
            unoccluded_tensor.unsqueeze(0), 
            size=target_size,
            # mode='nearest',
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        occluded_tensor = F.interpolate(
            occluded_tensor.unsqueeze(0), 
            size=target_size,
            # mode='nearest',
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        # Apply random hflip
        if random.random() < 0.5:
            unoccluded_tensor = TF.hflip(unoccluded_tensor)
            occluded_tensor = TF.hflip(occluded_tensor)

        # # Add subtle Gaussian noise to RGB sampled from N(0, noise_std)
        # noise = torch.randn_like(unoccluded_tensor) * self.noise_std
        # unoccluded_tensor = unoccluded_tensor + noise
        # occluded_tensor = occluded_tensor + noise
        # # Adding noise to a pixel value of 1.0 can push it to 1.02
        # unoccluded_tensor = torch.clamp(unoccluded_tensor, -1.0, 1.0)
        # occluded_tensor = torch.clamp(occluded_tensor, -1.0, 1.0)

        # Provide blank image for context-free guidance
        if random.random() < CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE:
            occluded_tensor = torch.full_like(occluded_tensor, -1.0)
        
        return unoccluded_tensor, occluded_tensor


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def get_pretrained_components(
        ) -> tuple[AutoencoderKL, UNet2DConditionModel, PNDMScheduler, CLIPTokenizer, CLIPTextModel]:
    print(f"Loading local models...")

    vae = AutoencoderKL.from_pretrained(vae_path)
    unet = UNet2DConditionModel.from_pretrained(unet_path)
    scheduler = PNDMScheduler.from_pretrained(scheduler_path)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
    
    return vae, unet, scheduler, tokenizer, text_encoder


def double_unet_input_capacity(
            unet: UNet2DConditionModel
        ) -> UNet2DConditionModel:
    with torch.no_grad():
        old_conv_in = unet.conv_in
        old_in_ch = old_conv_in.in_channels
        
        new_conv_in = nn.Conv2d(
            in_channels=2*old_in_ch, # 8 channels
            out_channels=old_conv_in.out_channels,
            kernel_size=old_conv_in.kernel_size,
            stride=old_conv_in.stride,
            padding=old_conv_in.padding
        )

        # Or, copy pretrained weights on both condition and noise with weights
        new_conv_in.weight[:, :old_in_ch, :, :] = old_conv_in.weight.clone() * (1 - UNET_LAYER1_WEIGHT_DIST_FOR_UNOCCLUDED) # occluded (condition)
        new_conv_in.weight[:, old_in_ch:, :, :] = old_conv_in.weight.clone() * UNET_LAYER1_WEIGHT_DIST_FOR_UNOCCLUDED # unoccluded (noise)
        
        new_conv_in.bias = nn.Parameter(old_conv_in.bias.clone())
        
        unet.conv_in = new_conv_in
        unet.register_to_config(in_channels=2*old_in_ch)
        
    return unet


def get_text_embedding(
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        text: str
        ) -> torch.Tensor:
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt" # Request PyTorch tensors ("pt")
    ).input_ids.to(text_encoder.device)

    with torch.no_grad():
        # embedding = text_encoder(tokens)[0]
        embedding = text_encoder(tokens).last_hidden_state

    print(f"embedding shape: {embedding.shape}")
    return embedding


def encode_image_batch_to_latents(
            vae: AutoencoderKL,
            image_batch: torch.Tensor,
        ) -> torch.Tensor:
    # print("Encoding image into Latent Space...")
    with torch.no_grad(): # Deactivates PyTorch's autograd engine, reducing unnecessary memory usage
        # Pass the tensor through the encoder to get the DiagonalGaussianDistribution
        latent_dist = vae.encode(image_batch).latent_dist
        
        # For deterministic encoding, we take the mode. 
        # Alternatively, you could use latent_dist.sample() for stochastic encoding.
        # latents = latent_dist.sample()
        latents = latent_dist.mode()
        
        # Stable Diffusion requires latents to be scaled by a specific factor
        scaling_factor = vae.config.scaling_factor
        latents = latents * scaling_factor

    # print(f"Original image shape: {image_batch.shape}")
    # print(f"Latent space shape: {latents.shape}") # Will be [1, 4, 64, 64] if input is 512x512
    
    return latents


def produceLogAndGraph(
            loss_log: list,
            model_path : str
        ) -> None:
    os.makedirs(model_path, exist_ok=True)
    
    with open(f"{model_path}/log.json", 'w+') as f:
        json.dump(loss_log, f, indent=4)
    
    plt.rcParams["font.family"] = "Serif"
    plt.figure(figsize=(16, 6))

    plt.title("Training MSE")
    plt.xlabel(f"Iterations (Every {PLT_DATA_SKIP}th)")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.plot(loss_log['train_mse'][::PLT_DATA_SKIP], label='Train MSE')
    # plt.plot(loss_log['epoch'], loss_log['test_ssim'], label='Test MSE')
    plt.legend()

    plt.savefig(f"{model_path}/graph.pdf", format="pdf", bbox_inches="tight")
    print(f"# graph saved in {model_path}")
    # plt.show()
    plt.clf()


def run_trainer(
            dataset: Dataset,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            out_model_path: str
        ) -> None:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=DATA_LOADER_WORKERS, # Adjust based on CPU cores
        pin_memory=True
    )
    
    device = get_device()
    
    vae, unet, scheduler, tokenizer, text_encoder = get_pretrained_components()
    vae = vae.to(device)
    vae.requires_grad_(False) # Freeze model weights and never compute gradients for them
    vae.eval() # Layer behavior adjustment, like- disable dropout layers & freeze batch normalization
    unet = double_unet_input_capacity(unet)
    unet = unet.to(device)
    unet.requires_grad_(True)
    scheduler = DDPMScheduler.from_config(scheduler.config)
    text_encoder = text_encoder.to(device)

    text_embedding = get_text_embedding(tokenizer, text_encoder, "").repeat(batch_size, 1, 1)
    # text_embedding = get_text_embedding(tokenizer, text_encoder, "").expand(batch_size, -1, -1) # less memory
    # text_embedding = get_zero_text_embedding(batch_size, unet.config.cross_attention_dim, device)
    
    # Free up VRAM by deleting the text encoder once the embedding is cached
    # del text_encoder, tokenizer 
    # torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=1e-2)

    loss_log = {
        'epoch': [],
        'train_mse': [],
    }
    print("Starting training...")
    for epoch in range(epochs):

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for unoccluded_rgbs, occluded_rgbs in progress_bar:
            unoccluded_latents = encode_image_batch_to_latents(vae, unoccluded_rgbs.to(device))
            occluded_latents = encode_image_batch_to_latents(vae, occluded_rgbs.to(device))
            
            # Sample random timesteps for each image in the batch
            timesteps = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps, 
                size=(batch_size,),
                device=device
            ).long()
            # Sample Gaussian noise
            noise = torch.randn_like(unoccluded_latents)
            # The timesteps dictates how much of that noise is blended into the clean latents
            noisy_unoccluded_latents = scheduler.add_noise(unoccluded_latents, noise, timesteps)

            unet_input = torch.cat([occluded_latents, noisy_unoccluded_latents], dim=1)
            noise_pred = unet(
                sample=unet_input, 
                timestep=timesteps, 
                encoder_hidden_states=text_embedding
            ).sample

            # Though we added scaled noise (depending on timesteps) to the image, we calculate
            # the loss against the original, unscaled noise. Because if we tried to predict the
            # scaled noise, the target values would be huge at step 999 and tiny at step 1. Deep
            # neural networks struggle to learn this. Whether the U-Net is looking at heavy noise
            # or light noise, the statistical properties of its target remain exactly the same.
            # This makes the loss gradient smooth and predictable.
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            # (Optional but highly recommended) Clip gradients to prevent random spikes
            # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            # Clear the old gradients before calculating new ones
            optimizer.zero_grad()
            # optimizer.zero_grad(set_to_none=True)
            
            loss_log['epoch'].append(epoch+1)
            loss_log['train_mse'].append(loss.item())
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    unet.save_pretrained(out_model_path)
    produceLogAndGraph(loss_log, out_model_path)
    print(f"Successfully saved model to: {out_model_path}")

    return 


if __name__ == "__main__":
    print("Process Started...")
    
    rgb_depth_instance_filepaths = list_rgb_depth_instance_in_hypersim(DATASET_ROOT)
    dataset = RGBCompletionDataset(rgb_depth_instance_filepaths, SD1_5_IMAGE_SIZE)
    run_trainer(dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, OUT_MODEL_PATH)

    print("Process Finished!")
