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
from pathlib import Path
from PIL import Image


vae_path = "vae_sd1-5_AutoencoderKL"
unet_path = "completion_sd1-5_5" #"unet_sd1-5_UNet2DConditionModel"
scheduler_path = "scheduler_sd1-5_PNDMScheduler"
tokenizer_path = "tokenizer_sd1-5_CLIPTokenizer"
text_encoder_path = "text_encoder_sd1-5_CLIPTextModel"

OUT_MODEL_PATH = "completion_sd1-5_6"

EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-5

# DATASET_ROOT = "/workspace/minhas/dataset/hypersim/unzips/"
DATASET_ROOT = "/workspace/minhas_dgx/hypersim_object_completion/"
SD1_5_IMAGE_SIZE = 512
DATA_LOADER_WORKERS = 2
DATA_DOWN_SAMPLE_CNT = -1 #10_000 #-1

PLT_DATA_SKIP = 100

UNET_LAYER1_WEIGHT_DIST_FOR_UNOCCLUDED = 0.5
CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE = 0.0


def list_paired_files(
            root_path: str
        ) -> list:
    print("Listing files")
    root = Path(root_path)
    occluded_dir = root / "occluded"
    unoccluded_dir = root / "unoccluded"

    if not occluded_dir.is_dir() or not unoccluded_dir.is_dir():
        raise FileNotFoundError(f"{occluded_dir} and/or {unoccluded_dir} directory does not exist.")

    print("occluded files...")
    occluded_files = {
        f.relative_to(occluded_dir): f
        for f in occluded_dir.rglob("*")
        if f.is_file()
    }

    print("unoccluded files...")
    unoccluded_files = {
        f.relative_to(unoccluded_dir): f
        for f in unoccluded_dir.rglob("*")
        if f.is_file()
    }

    common = sorted(occluded_files.keys() & unoccluded_files.keys())

    return [(unoccluded_files[p], occluded_files[p]) for p in common]


def load_img(
            path: str
        ) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img)


class RGBCompletionDataset(Dataset):
    def __init__(
                self, 
                in_out_filepaths: list,
                image_size: int,
            ) -> None:
        self.in_out_filepaths = in_out_filepaths
        self.image_size = image_size
        return

    def __len__(
                self
            ) -> int:
        return len(self.in_out_filepaths)

    def __getitem__(
                self,
                idx: int
            ) -> tuple[torch.Tensor, torch.Tensor]:

        unoccluded_path, occluded_path = self.in_out_filepaths[idx]
        unoccluded_rgb = load_img(unoccluded_path)
        occluded_rgb = load_img(occluded_path)

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
    # unet = double_unet_input_capacity(unet)
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
    print("Training...")
    
    file_pairs = list_paired_files(DATASET_ROOT)
    print(f"Number of training samples: {len(file_pairs)}")
    dataset = RGBCompletionDataset(file_pairs, SD1_5_IMAGE_SIZE)
    run_trainer(dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, OUT_MODEL_PATH)
