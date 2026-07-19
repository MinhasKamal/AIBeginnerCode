import os
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


vae_path = "vae_sd1-5_AutoencoderKL"
unet_path = "unet_sd1-5_UNet2DConditionModel"
scheduler_path = "scheduler_sd1-5_PNDMScheduler"
tokenizer_path = "tokenizer_sd1-5_CLIPTokenizer"
text_encoder_path = "text_encoder_sd1-5_CLIPTextModel"


class RGBDepthDataset(Dataset):
    def __init__(
                self, 
                rgb_dir: str,
                depth_dir: str,
                image_size: int,
                crop_len: int,
                noise_std: float
            ) -> None:
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        
        image_filenames_unsorted = [f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))]
        self.image_filenames = sorted(image_filenames_unsorted)

        self.image_size = image_size
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size+crop_len, image_size+crop_len)), # Resize to specific dimensions
            transforms.ToTensor(), # Changes from (H, W, Ch) to (Ch, H, W) & scales to [0.0, 1.0]
        ])
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalizes to [-1.0, 1.0] for the VAE

        self.crop_len = crop_len
        self.noise_std = noise_std

    def __len__(
                self
            ) -> int:
        return len(self.image_filenames)

    def __getitem__(
                self,
                idx
            ) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.image_filenames[idx]
        rgb_path = os.path.join(self.rgb_dir, filename)
        depth_path = os.path.join(self.depth_dir, filename)
        
        # Load and convert both to 3-channel RGB-like tensors
        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("RGB")

        # Convert to resized tensors
        rgb_tensor = self.base_transform(rgb_img)
        depth_tensor = self.base_transform(depth_img)

        # Random crop out
        top = random.randint(0, self.crop_len)
        left = random.randint(0, self.crop_len)
        rgb_tensor = TF.crop(rgb_tensor, top, left, self.image_size, self.image_size)
        depth_tensor = TF.crop(depth_tensor, top, left, self.image_size, self.image_size)

        # Apply random flip
        if random.random() > 0.5:
            rgb_tensor = TF.hflip(rgb_tensor)
            depth_tensor = TF.hflip(depth_tensor)

        # Add subtle Gaussian noise to RGB sampled from N(0, noise_std)
        noise = torch.randn_like(rgb_tensor) * self.noise_std
        rgb_tensor = rgb_tensor + noise
        # Adding noise to a pixel value of 1.0 can push it to 1.02
        rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)

        # Apply Normalization
        rgb_tensor = self.normalize(rgb_tensor)
        depth_tensor = self.normalize(depth_tensor)
        
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
        
        # Copy standard weights to RGB receiver channels [0:4]
        new_conv_in.weight[:, :old_in_ch, :, :] = old_conv_in.weight.clone() / 2
        
        # Zero-initialize the depth noise receiver channels [4:8]
        # new_conv_in.weight[:, old_in_ch:, :, :] = torch.zeros_like(new_conv_in.weight[:, old_in_ch:, :, :])
        # Or, copy from the old weights
        new_conv_in.weight[:, old_in_ch:, :, :] = old_conv_in.weight.clone() / 2
        
        new_conv_in.bias = nn.Parameter(old_conv_in.bias.clone())
        
        unet.conv_in = new_conv_in
        unet.register_to_config(in_channels=2*old_in_ch)
        
    return unet

def get_empty_text_embedding(
            batch_size: int,
            embedding_len: int,
            device: str
        ) -> torch.Tensor:
    maxCLIPTokenCount = 77
    text_embedding_shape = (batch_size, maxCLIPTokenCount, embedding_len)
    text_embedding = torch.zeros(text_embedding_shape, device=device)
    
    return text_embedding



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
    
    with open(f"{model_path}/log.json", 'w+') as f:
        json.dump(loss_log, f, indent=4)
    
    plt.rcParams["font.family"] = "Serif"
    plt.figure(figsize=(16, 6))

    plt.title('Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.plot(loss_log['epoch'], loss_log['train_mse'], label='Train MSE')
    # plt.plot(loss_log['epoch'], loss_log['test_ssim'], label='Test MSE')
    plt.legend()

    plt.savefig(f"{model_path}/graph.pdf", format="pdf", bbox_inches="tight")
    print(f"# graph saved in {model_path}")
    # plt.show()
    plt.clf()


def run_trainer(
            rgb_dir: str,
            depth_dir: str,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            out_model_path: str
        ) -> None:
    image_size = 512
    crop_len = 512 // 10
    noise_std = 0.02
    dataset = RGBDepthDataset(rgb_dir, depth_dir, image_size, crop_len, noise_std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    device = get_device()
    
    vae, unet, scheduler, tokenizer, text_encoder = get_pretrained_components()
    vae = vae.to(device)
    vae.requires_grad_(False) # Freeze model weights and never compute gradients for them
    vae.eval() # Layer behavior adjustment, like- disable dropout layers & freeze batch normalization
    unet = double_unet_input_capacity(unet)
    unet = unet.to(device)
    unet.requires_grad_(True)

    text_embedding = get_empty_text_embedding(batch_size, unet.config.cross_attention_dim, device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

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
            optimizer.step()

            loss_log['epoch'].append(epoch+1)
            loss_log['train_mse'].append(loss.item())
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    unet.save_pretrained(out_model_path)
    produceLogAndGraph(loss_log, out_model_path)
    print(f"Successfully saved model to: {out_model_path}")

    return 


if __name__ == "__main__":
    print("Process Started...")
    
    dataset_path = "/workspace/minhas/dataset/test_depth/"
    out_model_path = "depth_sd1-5"
    epochs = 300 #10
    batch_size = 6 #32
    learning_rate = 1e-5

    run_trainer(os.path.join(dataset_path, "rgb"), os.path.join(dataset_path, "depth"),
                epochs, batch_size, learning_rate, out_model_path)

    print("Process Finished!")
