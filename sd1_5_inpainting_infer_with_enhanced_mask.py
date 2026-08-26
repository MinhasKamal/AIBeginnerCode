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
from scipy.ndimage import binary_fill_holes


UNET_PATH = "inpainting_sd1-5_2"
IMAGE_SIZE = 512
INFERENCE_STEPS_LIST = [5, 10, 15, 20, 25, 30, 40, 60, 100, 150, 700]
GUIDANCE_SCALE_LIST = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 7.0]
IMAGE_PATH = "masked06.png"
DILATION_KERNEL_SIZE = 9


def get_device(
        ) -> str:
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


def get_rgba_tensor(
            image_path: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"image_path: {image_path}")
    img = Image.open(image_path).convert("RGBA")
    
    img_tensor = TF.to_tensor(img) # (H, W, Ch) -> (Ch, H, W) & scale to [0.0, 1.0]
    img_tensor = (img_tensor * 2 - 1.0) # [0.0, 1.0] -> [-1.0, 1.0]

    rgb_tensor = img_tensor[:3, :, :]
    alpha_tensor = img_tensor[3:, :, :]

    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
    # # So we unsqueeze(0) to fake a batch of 1 and interpolate
    rgb_tensor = F.interpolate(
        rgb_tensor.unsqueeze(0), 
        size=target_size,
        mode='nearest'
    )
    alpha_tensor = F.interpolate(
        alpha_tensor.unsqueeze(0), 
        size=target_size,
        mode='nearest'
    )
    
    return rgb_tensor, alpha_tensor


def dialate_mask(
            mask: torch.Tensor,
        ) -> torch.Tensor:

    kernel_size = DILATION_KERNEL_SIZE
    padding = kernel_size // 2
    dilated_mask = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )
    
    return dilated_mask


def enclose_mask(
            mask: torch.Tensor,
        ) -> torch.Tensor:
    mask_binary = (mask == 1).squeeze().cpu().numpy()
    mask_binary = binary_fill_holes(mask_binary)
    enclosed_mask = torch.where(torch.from_numpy(mask_binary), 1.0, -1.0).unsqueeze(0).unsqueeze(0)

    return enclosed_mask


def save_img_tensor(
            tensor: torch.Tensor,
            path: str,
        ):
    # Shift pixel values from [-1.0, 1.0] back to [0.0, 1.0]
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    
    # remove batch dimension -> bring to cpu -> Ch,H,W > H,W,Ch -> to numpy
    tensor = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    
    print(f"path: {path}")
    img.save(path)
    return

    
def save_img_tensor_grayscale(
            tensor: torch.Tensor,
            path: str,
        ):
    # Shift pixel values from [-1.0, 1.0] back to [0.0, 1.0]
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    
    # remove batch dimension -> bring to cpu -> Ch,H,W > H,W,Ch -> to numpy
    tensor = tensor.squeeze().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)

    img = Image.fromarray(tensor)
    
    print(f"path: {path}")
    img.save(path)
    return


def infer_classifier_free(
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            scheduler: PNDMScheduler,
            prompt_embeds: torch.Tensor,
            anti_prompt_embeds: torch.Tensor,
            rgb_tensor: torch.Tensor,
            mask_tensor: torch.Tensor,
            guidance_scale: float,
        ) -> torch.Tensor:
    print("Running classifier free...")
    blank_tensor = torch.zeros_like(rgb_tensor)
    # blank_tensor = torch.randn_like(rgb_tensor)
    
    with torch.no_grad():
        # Encode RGB into latents (4 channels)
        rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
        mask_latents = vae.encode(mask_tensor).latent_dist.mode() * vae.config.scaling_factor
        blank_latents = vae.encode(blank_tensor).latent_dist.mode() * vae.config.scaling_factor
        
        # Initialize pure random noise for the inpainted image (4 channels)
        inpainted_latents = torch.randn_like(rgb_latents)
        prompt_embeds = torch.cat([anti_prompt_embeds, prompt_embeds], dim=0)
        
        for t in scheduler.timesteps:
            # Concatenate all latents along the channel dimension
            # Shape becomes: [1, 12, 64, 64]
            rgb_input = torch.cat([rgb_latents, mask_latents, inpainted_latents], dim=1)
            blank_input = torch.cat([blank_latents, blank_latents, inpainted_latents], dim=1)
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
            # to produce the slightly cleaner inpainted latent for the next timestep
            inpainted_latents = scheduler.step(noise_pred, t, inpainted_latents).prev_sample
    
        ## 5. Decode Latents Back to Pixels
        # Un-scale the latents before decoding
        inpainted_latents = inpainted_latents / vae.config.scaling_factor
        inpainted_tensor = vae.decode(inpainted_latents).sample

    return inpainted_tensor


def inpaint_image(
            image_orig: torch.Tensor, 
            image_inpaint: torch.Tensor, 
            mask: torch.Tensor, 
        ) -> torch.Tensor:
    mask = (mask + 1.0) / 2.0
    image_inpainted = image_orig * (1.0 - mask) + image_inpaint * mask
    
    return image_inpainted


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
    anti_prompt_embeds = get_text_embedding(tokenizer, text_encoder, "")
    prompt_embeds = get_text_embedding(tokenizer, text_encoder, "")
    
    rgb_tensor, alpha_tensor = get_rgba_tensor(IMAGE_PATH)
    mask_tensor = alpha_tensor * -1
    
    save_img_tensor(rgb_tensor, f"{UNET_PATH}/{IMAGE_PATH}_rgb0.png")
    save_img_tensor_grayscale(alpha_tensor, f"{UNET_PATH}/{IMAGE_PATH}_alpha0.png")
    save_img_tensor_grayscale(mask_tensor, f"{UNET_PATH}/{IMAGE_PATH}_mask0.png")

    mask_tensor_enhanced = dialate_mask(mask_tensor)
    mask_tensor_enhanced = enclose_mask(mask_tensor_enhanced)

    mask_tensor_enhanced = mask_tensor_enhanced.repeat(1, 3, 1, 1) # [1, H, W] -> [3, H, W] -> [1, 3, H, W]
    rgb_tensor_eroded = rgb_tensor.clone()
    rgb_tensor_eroded[mask_tensor_enhanced > 0] = -1.0
    
    save_img_tensor(rgb_tensor_eroded, f"{UNET_PATH}/{IMAGE_PATH}_rgb.png")
    save_img_tensor(mask_tensor_enhanced, f"{UNET_PATH}/{IMAGE_PATH}_mask.png")

    rgb_tensor_eroded = rgb_tensor_eroded.to(device)
    mask_tensor_enhanced = mask_tensor_enhanced.to(device)

    for inference_steps in INFERENCE_STEPS_LIST:
        for guidance_scale in GUIDANCE_SCALE_LIST:
            scheduler.set_timesteps(inference_steps)
            output = infer_classifier_free(vae, unet, scheduler, prompt_embeds, anti_prompt_embeds, rgb_tensor_eroded, mask_tensor_enhanced, guidance_scale)
            save_img_tensor(output, f"{UNET_PATH}/{IMAGE_PATH}_{inference_steps}_{guidance_scale}.png")
            image_inpainted = inpaint_image(rgb_tensor, output.cpu(), mask_tensor)
            save_img_tensor(image_inpainted, f"{UNET_PATH}/{IMAGE_PATH}_{inference_steps}_{guidance_scale}_inpainted.png")
    
    print("Inference complete!")

