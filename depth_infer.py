import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import PNDMScheduler
from diffusers import DDPMScheduler


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

    
def get_empty_text_embedding(
            batch_size: int,
            embedding_len: int,
            device: str
        ) -> torch.Tensor:
    maxCLIPTokenCount = 77
    text_embedding_shape = (batch_size, maxCLIPTokenCount, embedding_len)
    text_embedding = torch.zeros(text_embedding_shape, device=device)
    
    return text_embedding


device = get_device()

## 1. Load Your Saved Models
vae_path = "vae_sd1-5_AutoencoderKL"
# unet_path = "unet_sd1-5_UNet2DConditionModel"
unet_path = "depth_sd1-5"
scheduler_path = "scheduler_sd1-5_PNDMScheduler"
vae = AutoencoderKL.from_pretrained(vae_path).to(device)
unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)
scheduler = PNDMScheduler.from_pretrained(scheduler_path)

# Set models to evaluation mode
vae.eval()
unet.eval()

## 2. Prepare the Input RGB Image
image_size = 512
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# image_path = "img.jpg"
image_path = "/workspace/minhas/dataset/test_depth/rgb/image_0001.png"
rgb_pil = Image.open(image_path).convert("RGB")
rgb_tensor = transform(rgb_pil).unsqueeze(0).to(device)

## 3. Encode RGB and Initialize Depth Noise
with torch.no_grad():
    # Encode RGB into latents (4 channels)
    rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
    
    # Initialize pure random noise for the depth map (4 channels)
    # This must be the exact same shape as the RGB latents
    depth_latents = torch.randn_like(rgb_latents)

## 4. The Reverse Diffusion (Denoising) Loop
inference_steps = 30
scheduler.set_timesteps(inference_steps)

# If your U-Net requires a prompt embedding (UNet2DConditionModel), 
# initialize an empty/dummy text embedding. 
# (If you used a standard UNet2DModel, you can remove this).
prompt_embeds = get_empty_text_embedding(1, unet.config.cross_attention_dim, device)

with torch.no_grad():
    for t in scheduler.timesteps:
        # Concatenate RGB latents and noisy depth latents along the channel dimension
        # Shape becomes: [1, 8, 64, 64]
        unet_input = torch.cat([rgb_latents, depth_latents], dim=1)
        
        # Predict the noise residual
        noise_pred = unet(
            sample=unet_input, 
            timestep=t, 
            encoder_hidden_states=prompt_embeds
        ).sample
        
        # Step the scheduler: removes a fraction of the predicted noise 
        # to produce the slightly cleaner depth latent for the next timestep
        depth_latents = scheduler.step(noise_pred, t, depth_latents).prev_sample

## 5. Decode Latents Back to Pixels
with torch.no_grad():
    # Un-scale the latents before decoding
    depth_latents = depth_latents / vae.config.scaling_factor
    depth_output = vae.decode(depth_latents).sample

## 6. Post-Process and Save the Depth Map
# Shift pixel values from [-1.0, 1.0] back to [0.0, 1.0]
depth_output = (depth_output / 2 + 0.5).clamp(0, 1)

# Convert tensor to a PIL Image
depth_output = depth_output.squeeze().cpu().permute(1, 2, 0).numpy()
depth_output = (depth_output * 255).astype(np.uint8)

depth_image = Image.fromarray(depth_output)

depth_image.save("predicted_depth_map.png")
print("Inference complete! Saved to predicted_depth_map.png")