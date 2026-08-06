import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import PNDMScheduler
from transformers import CLIPTokenizer
from transformers import CLIPTextModel


UNET_PATH = "completion_sd1-5_0"
IMAGE_SIZE = 512
INFERENCE_STEPS = 40
GUIDANCE_SCALE = 2.5
# IMAGE_PATH = "/workspace/minhas/dataset/test/3.jpg"
IMAGE_PATH = "compla1.png"
COMPLETION_PATH = "sd1_5_completion_pred.png"


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def get_pretrained_components(
            device: str
        ) -> tuple[AutoencoderKL, UNet2DConditionModel, PNDMScheduler, CLIPTokenizer, CLIPTextModel]:

    vae_path = "vae_sd1-5_AutoencoderKL"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    
    unet_path = UNET_PATH
    print(f"unet_path: {unet_path}")
    unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)
    
    scheduler_path = "scheduler_sd1-5_PNDMScheduler"
    scheduler = PNDMScheduler.from_pretrained(scheduler_path)
    
    tokenizer_path = "tokenizer_sd1-5_CLIPTokenizer"
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    
    text_encoder_path = "text_encoder_sd1-5_CLIPTextModel"
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(device)

    return vae, unet, scheduler, tokenizer, text_encoder


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

    # Free up VRAM by deleting the text encoder once the embedding is cached
    # del text_encoder, tokenizer 
    # torch.cuda.empty_cache()
    
    return embedding


def get_image_tensor(
            device: str
        ) -> torch.Tensor:
    ## Prepare the Input RGB Image
    print(f"image_path: {IMAGE_PATH}")
    rgb_img = Image.open(IMAGE_PATH).convert("RGB") # if RGB-A, then background is made black
    rgb_tensor = TF.to_tensor(rgb_img) # (H, W, Ch) -> (Ch, H, W) & scale to [0.0, 1.0]
    rgb_tensor = (rgb_tensor * 2 - 1.0) # [0.0, 1.0] -> [-1.0, 1.0]

    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
    # # So we unsqueeze(0) to fake a batch of 1, interpolate, and squeeze(0) to remove it
    rgb_tensor = F.interpolate(
        rgb_tensor.unsqueeze(0), 
        size=target_size,
        # mode='nearest',
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)

    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    
    return rgb_tensor


def save_image_tensor(
            rgb_tensor: torch.Tensor,
            completion_path: str
        ):
    # Shift pixel values from [-1.0, 1.0] back to [0.0, 1.0]
    rgb_tensor = (rgb_tensor / 2 + 0.5).clamp(0, 1)
    
    # Convert tensor to a PIL Image
    rgb_tensor = rgb_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    rgb_tensor = (rgb_tensor * 255).astype(np.uint8)
    
    rgb_image = Image.fromarray(rgb_tensor)
    
    print(f"completion path: {completion_path}")
    rgb_image.save(completion_path)
    return


def infer(
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            scheduler: PNDMScheduler,
            prompt_embeds: torch.Tensor,
            occluded_tensor: torch.Tensor,
        ) -> torch.Tensor:

    with torch.no_grad():
        # Encode occluded img into latents (4 channels)
        occluded_latents = vae.encode(occluded_tensor).latent_dist.mode() * vae.config.scaling_factor
        
        # Initialize pure random noise for the unoccluded img (4 channels)
        # This must be the exact same shape as the occluded latents
        unoccluded_latents = torch.randn_like(occluded_latents)
        
        for t in scheduler.timesteps:
            # Concatenate occluded latents and noisy unoccluded latents along the channel dimension
            # Shape becomes: [1, 8, 64, 64]
            unet_input = torch.cat([occluded_latents, unoccluded_latents], dim=1)
            
            # Predict the noise residual
            noise_pred = unet(
                sample=unet_input,
                timestep=t, 
                encoder_hidden_states=prompt_embeds
            ).sample
            
            # Step the scheduler: removes a fraction of the predicted noise 
            # to produce the slightly cleaner unoccluded latent for the next timestep
            unoccluded_latents = scheduler.step(noise_pred, t, unoccluded_latents).prev_sample
    
        ## 5. Decode Latents Back to Pixels
        # Un-scale the latents before decoding
        unoccluded_latents = unoccluded_latents / vae.config.scaling_factor
        unoccluded_output = vae.decode(unoccluded_latents).sample

    return unoccluded_output


def infer_classifier_free(
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        prompt_embeds: torch.Tensor,
        occluded_tensor: torch.Tensor,
        guidance_scale: float = GUIDANCE_SCALE,
        ) -> torch.Tensor:
    print("Running classifier free...")
    blank_tensor = torch.zeros_like(occluded_tensor)

    with torch.no_grad():
        # Encode occluded img into latents (4 channels)
        occluded_latents = vae.encode(occluded_tensor).latent_dist.mode() * vae.config.scaling_factor
        blank_latents = vae.encode(blank_tensor).latent_dist.mode() * vae.config.scaling_factor
        
        # Initialize pure random noise for the unoccluded img (4 channels)
        # This must be the exact same shape as the occluded latents
        unoccluded_latents = torch.randn_like(occluded_latents)
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        
        for t in scheduler.timesteps:
            # Concatenate occluded latents and noisy unoccluded latents along the channel dimension
            # Shape becomes: [1, 8, 64, 64]
            occluded_input = torch.cat([occluded_latents, unoccluded_latents], dim=1)
            blank_input = torch.cat([blank_latents, unoccluded_latents], dim=1)
            unet_input = torch.cat([blank_input, occluded_input], dim=0)
            
            # Predict the noise residual
            noise_pred = unet(
                sample=unet_input,
                timestep=t, 
                encoder_hidden_states=prompt_embeds
            ).sample

            # Apply Classifier-Free Guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Step the scheduler: removes a fraction of the predicted noise 
            # to produce the slightly cleaner unoccluded latent for the next timestep
            unoccluded_latents = scheduler.step(noise_pred, t, unoccluded_latents).prev_sample
    
        ## 5. Decode Latents Back to Pixels
        # Un-scale the latents before decoding
        unoccluded_latents = unoccluded_latents / vae.config.scaling_factor
        unoccluded_tensor = vae.decode(unoccluded_latents).sample

    return unoccluded_tensor


if __name__ == "__main__":
    print("Inferring...")
    
    device = get_device()
    
    ## Load Your Saved Models
    vae, unet, scheduler, tokenizer, text_encoder = get_pretrained_components(device)
    # Set models to evaluation mode
    vae.eval()
    unet.eval()
    text_encoder.eval()
    
    ## The reverse diffusion (denoising) loop steps
    scheduler.set_timesteps(INFERENCE_STEPS)
    
    # Initialize an empty text embedding
    prompt_embeds = get_text_embedding(tokenizer, text_encoder, "")
    
    occluded_tensor = get_image_tensor(device)
    save_image_tensor(occluded_tensor, COMPLETION_PATH+"in.png")
    
    unoccluded_tensor = infer(vae, unet, scheduler, prompt_embeds, occluded_tensor)
    # unoccluded_tensor = infer_classifier_free(vae, unet, scheduler, prompt_embeds, occluded_tensor)
    
    save_image_tensor(unoccluded_tensor, COMPLETION_PATH)
    
    print("Inference complete!")

