import os
import glob
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import PNDMScheduler
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer
from transformers import CLIPTextModel



DATASET_ROOT = "/workspace/minhas/dataset/hypersim/unzips/"
DATA_DOWN_SAMPLE_CNT = -1 #10_000 #-1
DATA_LOADER_WORKERS = 2

SD1_5_IMAGE_SIZE = 512
MAX_MASK_INSTANCE_COUNT = 5

vae_path = "vae_sd1-5_AutoencoderKL"
unet_path = "unet_sd1-5_UNet2DConditionModel"
scheduler_path = "scheduler_sd1-5_PNDMScheduler"
tokenizer_path = "tokenizer_sd1-5_CLIPTokenizer"
text_encoder_path = "text_encoder_sd1-5_CLIPTextModel"

OUT_MODEL_PATH = "inpainting_sd1-5_1"

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-5

PLT_DATA_SKIP = 100



def plot_tensor_list(
            tensor_list: list,
            file_name: str
        ):
    np_arr_list = [t.detach().cpu().permute(1, 2, 0).numpy() for t in tensor_list]
    np_arr_list = [(a+1)/2 for a in np_arr_list]
    plot_np_arr_list(np_arr_list, file_name)
    return
    

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



##########################################################################

def list_rgb_depth_instance_in_hypersim(
            dataset_root: str
        ) -> list:
    print("Listing rgb-depth-instance...")
    grouped_filepaths = []

    # Get all scene directories (e.g., ai_001_001, ai_002_001)
    scene_dirs_pattern = os.path.join(dataset_root, "ai_*")
    scene_dirs = glob.glob(scene_dirs_pattern)
    
    for scene_dir in tqdm(sorted(scene_dirs)):
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
                # break ##############################################################################################

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


def read_instance_mask_in_hypersim(
            instance_mask_path: str
        ) -> np.ndarray:
    with h5py.File(instance_mask_path, 'r') as f:
        instance_mask = np.array(f["dataset"], dtype=np.int32)

    # print(f"instance: {instance_mask.shape}, min: {instance_mask.min()}, max: {instance_mask.max()}")
    return instance_mask


##########################################################################

def get_random_except(start, end, excluded) -> int:
    while True:
        val = random.randint(start, end)
        if val != excluded:
            return val


def get_random_mask(
            grouped_filepaths: list,
            data_index: int,
        ) -> np.ndarray:
    while True:
        random_mask_index = get_random_except(0, len(grouped_filepaths)-1, data_index)
        instance_mask = read_instance_mask_in_hypersim(grouped_filepaths[random_mask_index][2])
    
        instance_ids = np.unique(instance_mask)
        if len(instance_ids) < 2:
            # print(f"! instance_ids: {instance_ids}")
            continue

        mask_index = random.randint(1, len(instance_ids) - 1) 
        mask = (instance_mask == instance_ids[mask_index])
    
        mask_size_ratio = np.count_nonzero(mask) / mask.size
    
        if mask_size_ratio < 0.001 or mask_size_ratio > 0.25:
            # print(f"! mask_size_ratio: {mask_size_ratio}")
            continue

        break

    return mask


def generate_one_random_item(
            grouped_filepaths: list,
            data_index: int,
            max_mask_instance_count: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exposure_value = random.uniform(-1.0, 1.0)

    full_rgb = read_rgb_img_in_hypersim(grouped_filepaths[data_index][0], exposure_value)

    mask_instance_count = random.randint(1, max_mask_instance_count+1)
    mask = np.zeros((full_rgb.shape[0], full_rgb.shape[1]), dtype=bool)
    for i in range(mask_instance_count):
        mask += get_random_mask(grouped_filepaths, data_index)

    masked_rgb = np.zeros_like(full_rgb)
    alpha = (mask == False)
    masked_rgb[alpha] = full_rgb[alpha]
    
    return full_rgb, mask, masked_rgb


##########################################################################


class RGBInpaintingDataset(Dataset):
    def __init__(
                self, 
                rgb_depth_instance_filepaths: list,
                image_size: int,
                max_mask_instance_count: int,
            ) -> None:
        self.rgb_depth_instance_filepaths = rgb_depth_instance_filepaths
        self.image_size = image_size
        self.max_mask_instance_count = max_mask_instance_count
        return

    def __len__(
                self
            ) -> int:
        return len(self.rgb_depth_instance_filepaths)

    def __getitem__(
                self,
                idx: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        full_img, mask, masked_img = generate_one_random_item(self.rgb_depth_instance_filepaths, idx, self.max_mask_instance_count)

        full_img_tensor = torch.from_numpy(full_img).float()
        mask_tensor = torch.from_numpy(mask).float()
        masked_img_tensor = torch.from_numpy(masked_img).float()


        # (H, W, Ch) -> (Ch, H, W)
        full_img_tensor = full_img_tensor.permute(2, 0, 1)
        masked_img_tensor = masked_img_tensor.permute(2, 0, 1)
        
        mask_tensor = mask_tensor.unsqueeze(0) # [H, W] -> [1, H, W]
        mask_tensor = mask_tensor.repeat(3, 1, 1) # [1, H, W] -> [3, H, W]

        # scale to [-1.0, 1.0]
        full_img_tensor = ((full_img_tensor / 255.0) * 2.0) - 1.0
        mask_tensor = (mask_tensor * 2.0) - 1.0
        masked_img_tensor = ((masked_img_tensor / 255.0) * 2.0) - 1.0

        target_size = (self.image_size, self.image_size)
        # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
        # # So we unsqueeze(0) to fake a batch of 1, interpolate, and squeeze(0) to remove it
        full_img_tensor = F.interpolate(
            full_img_tensor.unsqueeze(0), 
            size=target_size,
            mode='nearest'
        ).squeeze(0)
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0), 
            size=target_size,
            mode='nearest'
        ).squeeze(0)
        masked_img_tensor = F.interpolate(
            masked_img_tensor.unsqueeze(0), 
            size=target_size,
            mode='nearest'
        ).squeeze(0)

        # Apply random hflip
        if random.random() < 0.5:
            full_img_tensor = TF.hflip(full_img_tensor)
            mask_tensor = TF.hflip(mask_tensor)
            masked_img_tensor = TF.hflip(masked_img_tensor)

        # # Add subtle Gaussian noise to RGB sampled from N(0, noise_std)
        # noise = torch.randn_like(unoccluded_tensor) * self.noise_std
        # unoccluded_tensor = unoccluded_tensor + noise
        # occluded_tensor = occluded_tensor + noise
        # # Adding noise to a pixel value of 1.0 can push it to 1.02
        # unoccluded_tensor = torch.clamp(unoccluded_tensor, -1.0, 1.0)
        # occluded_tensor = torch.clamp(occluded_tensor, -1.0, 1.0)

        # Provide blank image for context-free guidance
        # if random.random() < CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE:
        #     occluded_tensor = torch.full_like(occluded_tensor, -1.0)
        
        return full_img_tensor, mask_tensor, masked_img_tensor


##########################################################################


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


def multiply_unet_input_capacity(
            unet: UNet2DConditionModel,
            times: int,
        ) -> UNet2DConditionModel:
    with torch.no_grad():
        old_conv_in = unet.conv_in
        old_in_ch = old_conv_in.in_channels
        
        new_conv_in = nn.Conv2d(
            in_channels=times*old_in_ch, # channels
            out_channels=old_conv_in.out_channels,
            kernel_size=old_conv_in.kernel_size,
            stride=old_conv_in.stride,
            padding=old_conv_in.padding
        )

        # copy pretrained weights on both condition and noise
        weight_ratio = 1.0 / times
        for i in range(times):
            start_ch = i * old_in_ch
            end_ch = (i + 1) * old_in_ch
            new_conv_in.weight[:, start_ch:end_ch, :, :] = old_conv_in.weight.clone() * weight_ratio

        new_conv_in.bias = nn.Parameter(old_conv_in.bias.clone())
        
        unet.conv_in = new_conv_in
        unet.register_to_config(in_channels=times*old_in_ch)
        
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

    print(f"text embedding shape: {embedding.shape}")
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
    unet = multiply_unet_input_capacity(unet, 3)
    unet = unet.to(device)
    unet.requires_grad_(True)
    scheduler = DDPMScheduler.from_config(scheduler.config)
    text_encoder = text_encoder.to(device)

    text_embedding = get_text_embedding(tokenizer, text_encoder, "").repeat(batch_size, 1, 1)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=1e-2)

    loss_log = {
        'epoch': [],
        'train_mse': [],
    }
    print("Starting training...")
    for epoch in range(epochs):

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for full_imgs, masks, masked_imgs in progress_bar:
            full_img_latents = encode_image_batch_to_latents(vae, full_imgs.to(device))
            mask_latents = encode_image_batch_to_latents(vae, masks.to(device))
            masked_img_latents = encode_image_batch_to_latents(vae, masked_imgs.to(device))
            
            # Sample random timesteps for each image in the batch
            timesteps = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps, 
                size=(batch_size,),
                device=device
            ).long()
            # Sample Gaussian noise
            noise = torch.randn_like(full_img_latents)
            # The timesteps dictates how much of that noise is blended into the clean latents
            noisy_full_img_latents = scheduler.add_noise(full_img_latents, noise, timesteps)

            unet_input = torch.cat([masked_img_latents, mask_latents, noisy_full_img_latents], dim=1)
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
    print("Training started...")
    
    rgb_depth_instance_filepaths = list_rgb_depth_instance_in_hypersim(DATASET_ROOT)
    dataset = RGBInpaintingDataset(rgb_depth_instance_filepaths, SD1_5_IMAGE_SIZE, MAX_MASK_INSTANCE_COUNT)
    full_img_tensor, mask_tensor, masked_img_tensor = dataset.__getitem__(0)
    plot_tensor_list([full_img_tensor, mask_tensor, masked_img_tensor], "sd1_5_inpainting.png")
    run_trainer(dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, OUT_MODEL_PATH)


    