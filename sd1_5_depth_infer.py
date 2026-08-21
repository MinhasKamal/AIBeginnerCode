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
import cv2


UNET_PATH = "depth_sd1-5_3"
IMAGE_SIZE = 512
# INFERENCE_STEPS = 200 # 25
# GUIDANCE_SCALE = 3.0 # 2.5
INFERENCE_STEPS_LIST = [15, 20, 25, 30, 40, 60, 100, 150, 250, 500, 990]
GUIDANCE_SCALE_LIST = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 7.0, 10.0]
# IMAGE_PATH = "/workspace/minhas/dataset/test_depth/rgb/image_0001.png"
# IMAGE_PATH = "/workspace/minhas/dataset/test/3.jpg"
IMAGE_PATH = "rgb05.png"
# DEPTH_PATH = "sd1_5_depth_pred.png"


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def get_pretrained_components(
        device: str
        ) -> tuple[AutoencoderKL, UNet2DConditionModel, PNDMScheduler, CLIPTokenizer, CLIPTextModel]:

    vae_path = "vae_sd1-5_AutoencoderKL"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    
    # unet_path = "unet_sd1-5_UNet2DConditionModel"
    unet_path = UNET_PATH
    # unet_path = "depth_sd1-5_4"
    print(f"unet_path: {unet_path}")
    unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)
    
    scheduler_path = "scheduler_sd1-5_PNDMScheduler"
    scheduler = PNDMScheduler.from_pretrained(scheduler_path)
    
    tokenizer_path = "tokenizer_sd1-5_CLIPTokenizer"
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    
    text_encoder_path = "text_encoder_sd1-5_CLIPTextModel"
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(device)

    return vae, unet, scheduler, tokenizer, text_encoder


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

    # Free up VRAM by deleting the text encoder once the embedding is cached
    # del text_encoder, tokenizer 
    # torch.cuda.empty_cache()
    
    return embedding


def get_image_tensor(
        device: str
        ) -> torch.Tensor:
    ## Prepare the Input RGB Image
    print(f"image_path: {IMAGE_PATH}")
    rgb_img = Image.open(IMAGE_PATH).convert("RGB")
    rgb_tensor = TF.to_tensor(rgb_img) # (H, W, Ch) -> (Ch, H, W) & scale to [0.0, 1.0]
    rgb_tensor = (rgb_tensor * 2 - 1.0) # [0.0, 1.0] -> [-1.0, 1.0]

    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
    # # So we unsqueeze(0) to fake a batch of 1, interpolate, and squeeze(0) to remove it
    rgb_tensor = F.interpolate(
        rgb_tensor.unsqueeze(0), 
        size=target_size,
        mode='nearest',
        # mode='bilinear', 
        # align_corners=False
    ).squeeze(0)

    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    
    return rgb_tensor

    
def save_image_tensor_colormap(
        depth_output: torch.Tensor,
        depth_path: str,
        ):
    # Shift pixel values from [-1.0, 1.0] back to [0.0, 1.0]
    depth_output = (depth_output / 2 + 0.5).clamp(0, 1)
    
    # remove batch dimension -> bring to cpu -> Ch,H,W > H,W,Ch -> to numpy
    depth_output = depth_output.squeeze()[0].detach()
    depth_output = depth_output.cpu().numpy()
    depth_output = (depth_output * 255).astype(np.uint8)

    inverted_depth_output = 255 - depth_output

    rgb_depth = cv2.applyColorMap(inverted_depth_output, cv2.COLORMAP_JET) # VIRIDIS, PLASMA
    
    print(f"depth_path: {depth_path}")
    cv2.imwrite(depth_path, rgb_depth)
    return


def save_image_tensor(
        depth_output: torch.Tensor,
        depth_path: str,
        ):
    # Shift pixel values from [-1.0, 1.0] back to [0.0, 1.0]
    depth_output = (depth_output / 2 + 0.5).clamp(0, 1)
    
    # remove batch dimension -> bring to cpu -> Ch,H,W > H,W,Ch -> to numpy
    depth_output = depth_output.squeeze().cpu().permute(1, 2, 0).numpy()
    depth_output = (depth_output * 255).astype(np.uint8)
    
    depth_image = Image.fromarray(depth_output)
    
    print(f"depth_path: {depth_path}")
    depth_image.save(depth_path)
    return


def infer(
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        prompt_embeds: torch.Tensor,
        rgb_tensor: torch.Tensor,
        ) -> torch.Tensor:

    with torch.no_grad():
        # Encode RGB into latents (4 channels)
        rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
        
        # Initialize pure random noise for the depth map (4 channels)
        # This must be the exact same shape as the RGB latents
        depth_latents = torch.randn_like(rgb_latents)
        
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
        # Un-scale the latents before decoding
        depth_latents = depth_latents / vae.config.scaling_factor
        depth_output = vae.decode(depth_latents).sample

    return depth_output


def infer_classifier_free(
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        prompt_embeds: torch.Tensor,
        rgb_tensor: torch.Tensor,
        guidance_scale: float,
        ) -> torch.Tensor:
    print("Running classifier free...")
    blank_tensor = torch.zeros_like(rgb_tensor)
    
    with torch.no_grad():
        # Encode RGB into latents (4 channels)
        rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
        blank_latents = vae.encode(blank_tensor).latent_dist.mode() * vae.config.scaling_factor
        
        # Initialize pure random noise for the depth map (4 channels)
        # This must be the exact same shape as the RGB latents
        depth_latents = torch.randn_like(rgb_latents)
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        
        for t in scheduler.timesteps:
            # Concatenate RGB latents and noisy depth latents along the channel dimension
            # Shape becomes: [1, 8, 64, 64]
            rgb_input = torch.cat([rgb_latents, depth_latents], dim=1)
            blank_input = torch.cat([blank_latents, depth_latents], dim=1)
            unet_input = torch.cat([blank_input, rgb_input], dim=0)
            
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
            # to produce the slightly cleaner depth latent for the next timestep
            depth_latents = scheduler.step(noise_pred, t, depth_latents).prev_sample
    
        ## 5. Decode Latents Back to Pixels
        # Un-scale the latents before decoding
        depth_latents = depth_latents / vae.config.scaling_factor
        depth_output = vae.decode(depth_latents).sample

    return depth_output

    
if __name__ == "__main__":
    print("Started...")
    
    device = get_device()
    
    ## Load Your Saved Models
    vae, unet, scheduler, tokenizer, text_encoder = get_pretrained_components(device)
    # Set models to evaluation mode
    vae.eval()
    unet.eval()
    text_encoder.eval()
    
    
    # Initialize an empty text embedding
    prompt_embeds = get_text_embedding(tokenizer, text_encoder, "")
    # prompt_embeds = get_zero_text_embedding(1, unet.config.cross_attention_dim, device)
    
    rgb_tensor = get_image_tensor(device)
    save_image_tensor(rgb_tensor, f"{UNET_PATH}/{IMAGE_PATH}_in.png")
    
    # scheduler.set_timesteps(INFERENCE_STEPS)
    # # depth_output = infer(vae, unet, scheduler, prompt_embeds, rgb_tensor)
    # depth_output = infer_classifier_free(vae, unet, scheduler, prompt_embeds, rgb_tensor, GUIDANCE_SCALE)
    # save_image_tensor(depth_output, DEPTH_PATH)

    for inference_steps in INFERENCE_STEPS_LIST:
        for guidance_scale in GUIDANCE_SCALE_LIST:
            scheduler.set_timesteps(inference_steps)
            depth_output = infer_classifier_free(vae, unet, scheduler, prompt_embeds, rgb_tensor, guidance_scale)
            save_image_tensor_colormap(depth_output, f"{UNET_PATH}/{IMAGE_PATH}_{inference_steps}_{guidance_scale}.png")
    
    print("Inference complete!")

