# Minhas Kamal
# 20 Jul 2026
# Requires 80GB VRAM


import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import PNDMScheduler
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer
from transformers import CLIPTextModel
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import h5py


vae_path = "vae_sd1-5_AutoencoderKL"
unet_path = "unet_sd1-5_UNet2DConditionModel"
scheduler_path = "scheduler_sd1-5_PNDMScheduler"
tokenizer_path = "tokenizer_sd1-5_CLIPTokenizer"
text_encoder_path = "text_encoder_sd1-5_CLIPTextModel"
OUT_MODEL_PATH = "depth_sd1-5_3"

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-5

SD1_5_IMAGE_SIZE = 512
IMG_AUG_CROP_LEN = SD1_5_IMAGE_SIZE // 10
IMG_AUG_NOISE_STD = 0.02
MAX_DEPTH_METERS = 25.0

DATASET_ROOT = "/workspace/minhas/dataset/hypersim/unzips/"
DATA_DOWN_SAMPLE_CNT = -1
PLT_DATA_SKIP = 100

UNET_LAYER1_WEIGHT_DIST_FOR_DEPTH = 0.5
CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE = 0.2


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
                    
    if DATA_DOWN_SAMPLE_CNT > 0:
        paired_filepaths = random.sample(paired_filepaths, DATA_DOWN_SAMPLE_CNT)
    
    print(f"Total rgb-depth frame pairs {len(paired_filepaths)}")
    # print(paired_filepaths)
    return paired_filepaths


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


#####################################################################################################


class RGBDepthDataset(Dataset):
    def __init__(
                self, 
                paired_filepaths: list,
                image_size: int,
                crop_len: int,
                noise_std: float
            ) -> None:
        self.paired_filepaths = paired_filepaths
        self.image_size = image_size
        self.crop_len = crop_len
        self.noise_std = noise_std
        return

    def __len__(
                self
            ) -> int:
        return len(self.paired_filepaths)

    def __getitem__(
                self,
                idx: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_path, depth_path = self.paired_filepaths[idx]
        
        # Load and convert both to 3-channel RGB-like tensors
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_tensor = TF.to_tensor(rgb_img) # (H, W, Ch) -> (Ch, H, W) & scale to [0.0, 1.0]
        rgb_tensor = (rgb_tensor * 2 - 1.0) # [0.0, 1.0] -> [-1.0, 1.0]
        depth_tensor = read_depth_in_hypersim(depth_path)

        target_size = (self.image_size + self.crop_len, self.image_size + self.crop_len)
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

        # Random crop out
        top = random.randint(0, self.crop_len)
        left = random.randint(0, self.crop_len)
        rgb_tensor = TF.crop(rgb_tensor, top, left, self.image_size, self.image_size)
        depth_tensor = TF.crop(depth_tensor, top, left, self.image_size, self.image_size)

        # Apply random hflip
        if random.random() < 0.5:
            rgb_tensor = TF.hflip(rgb_tensor)
            depth_tensor = TF.hflip(depth_tensor)

        # Add subtle Gaussian noise to RGB sampled from N(0, noise_std)
        noise = torch.randn_like(rgb_tensor) * self.noise_std
        rgb_tensor = rgb_tensor + noise
        # Adding noise to a pixel value of 1.0 can push it to 1.02
        rgb_tensor = torch.clamp(rgb_tensor, -1.0, 1.0)

        # Provide blank RGB image for context-free guidance
        if random.random() < CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE:
            rgb_tensor = torch.zeros_like(rgb_tensor)
        
        return rgb_tensor, depth_tensor


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
        
        # Zero-initialize the rgb channels [0:4] & Copy from the old weights for the depth channel [4:8]
        # new_conv_in.weight[:, :old_in_ch, :, :] = torch.zeros_like(new_conv_in.weight[:, :old_in_ch, :, :])
        # new_conv_in.weight[:, old_in_ch:, :, :] = old_conv_in.weight.clone()
        # Or, copy pretrained weights on both rgb and depth with weights
        new_conv_in.weight[:, :old_in_ch, :, :] = old_conv_in.weight.clone() * (1 - UNET_LAYER1_WEIGHT_DIST_FOR_DEPTH) # RGB
        new_conv_in.weight[:, old_in_ch:, :, :] = old_conv_in.weight.clone() * UNET_LAYER1_WEIGHT_DIST_FOR_DEPTH # Depth
        
        new_conv_in.bias = nn.Parameter(old_conv_in.bias.clone())
        
        unet.conv_in = new_conv_in
        unet.register_to_config(in_channels=2*old_in_ch)
        
    return unet


def get_zero_text_embedding(
            batch_size: int,
            embedding_len: int,
            device: str
        ) -> torch.Tensor:
    maxCLIPTokenCount = 77
    text_embedding_shape = (batch_size, maxCLIPTokenCount, embedding_len)
    text_embedding = torch.zeros(text_embedding_shape, device=device)

    print(f"embedding shape: {text_embedding.shape}")
    return text_embedding


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
        num_workers=2, # Adjust based on CPU cores
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
        for rgb_imgs, depth_imgs in progress_bar:
            rgb_latents = encode_image_batch_to_latents(vae, rgb_imgs.to(device))
            depth_latents = encode_image_batch_to_latents(vae, depth_imgs.to(device))
            
            # Sample random timesteps for each image in the batch
            timesteps = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps, 
                size=(batch_size,),
                device=device
            ).long()
            # Sample Gaussian noise
            noise = torch.randn_like(depth_latents)
            # The timesteps dictates how much of that noise is blended into the clean latents
            noisy_depth_latents = scheduler.add_noise(depth_latents, noise, timesteps)

            unet_input = torch.cat([rgb_latents, noisy_depth_latents], dim=1)
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
    
    paired_filepaths = list_rgb_depth_pairs_in_hypersim(DATASET_ROOT)
    dataset = RGBDepthDataset(paired_filepaths, SD1_5_IMAGE_SIZE, IMG_AUG_CROP_LEN, IMG_AUG_NOISE_STD)
    run_trainer(dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, OUT_MODEL_PATH)

    print("Process Finished!")
