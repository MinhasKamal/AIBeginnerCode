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


UNET_PATH = "inpainting_sd1-5_2"
IMAGE_SIZE = 512
INFERENCE_STEPS_LIST = [15, 20, 25, 30, 40, 60, 100, 150, 250, 500, 990]
GUIDANCE_SCALE_LIST = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 7.0, 10.0]
IMAGE_PATH = "masked04.png"


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
            device: str
        ) -> torch.Tensor:
    print(f"image_path: {image_path}")
    img = Image.open(image_path).convert("RGBA")
    
    img_tensor = TF.to_tensor(img) # (H, W, Ch) -> (Ch, H, W) & scale to [0.0, 1.0]
    img_tensor = (img_tensor * 2 - 1.0) # [0.0, 1.0] -> [-1.0, 1.0]

    rgb_tensor = img_tensor[:3, :, :]
    alpha_tensor = img_tensor[3:, :, :]
    alpha_tensor = alpha_tensor.repeat(3, 1, 1) # [1, H, W] -> [3, H, W]

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

    # rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    rgb_tensor = rgb_tensor.to(device)
    alpha_tensor = alpha_tensor.to(device)
    
    return rgb_tensor, alpha_tensor


# def save_image_tensor_colormap(
#         depth_output: torch.Tensor,
#         depth_path: str,
#         ):
#     # Shift pixel values from [-1.0, 1.0] back to [0.0, 1.0]
#     depth_output = (depth_output / 2 + 0.5).clamp(0, 1)
    
#     # remove batch dimension -> bring to cpu -> Ch,H,W > H,W,Ch -> to numpy
#     depth_output = depth_output.squeeze()[0].detach()
#     depth_output = depth_output.cpu().numpy()
#     depth_output = (depth_output * 255).astype(np.uint8)

#     inverted_depth_output = 255 - depth_output

#     rgb_depth = cv2.applyColorMap(inverted_depth_output, cv2.COLORMAP_JET) # VIRIDIS, PLASMA
    
#     print(f"depth_path: {depth_path}")
#     cv2.imwrite(depth_path, rgb_depth)
#     return


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


def infer_classifier_free(
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            scheduler: PNDMScheduler,
            prompt_embeds: torch.Tensor,
            blank_prompt_embeds: torch.Tensor,
            rgb_tensor: torch.Tensor,
            mask_tensor: torch.Tensor,
            guidance_scale: float,
        ) -> torch.Tensor:
    print("Running classifier free...")
    blank_tensor = torch.zeros_like(rgb_tensor)
    
    with torch.no_grad():
        # Encode RGB into latents (4 channels)
        rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
        mask_latents = vae.encode(mask_tensor).latent_dist.mode() * vae.config.scaling_factor
        blank_latents = vae.encode(blank_tensor).latent_dist.mode() * vae.config.scaling_factor
        
        # Initialize pure random noise for the inpainted image (4 channels)
        inpainted_latents = torch.randn_like(rgb_latents)
        prompt_embeds = torch.cat([blank_prompt_embeds, prompt_embeds], dim=0)
        
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
    blank_prompt_embeds = get_text_embedding(tokenizer, text_encoder, "Create new object and pattern.")
    prompt_embeds = get_text_embedding(tokenizer, text_encoder, "Inpaint the missing area using only the surrounding background- floor and wall. Match the existing background naturally. Do not add or create any object or new patterns. Do not generate new objects or foreground content. The result should contain only the background.")
    
    rgb_tensor, alpha_tensor = get_rgba_tensor(IMAGE_PATH, device)
    mask_tensor = alpha_tensor * -1
    
    save_img_tensor(rgb_tensor, f"{UNET_PATH}/{IMAGE_PATH}_rgb.png")
    save_img_tensor(alpha_tensor, f"{UNET_PATH}/{IMAGE_PATH}_alpha.png")
    save_img_tensor(mask_tensor, f"{UNET_PATH}/{IMAGE_PATH}_mask.png")

    for inference_steps in INFERENCE_STEPS_LIST:
        for guidance_scale in GUIDANCE_SCALE_LIST:
            scheduler.set_timesteps(inference_steps)
            output = infer_classifier_free(vae, unet, scheduler, prompt_embeds, blank_prompt_embeds, rgb_tensor, mask_tensor, guidance_scale)
            save_img_tensor(output, f"{UNET_PATH}/{IMAGE_PATH}_{inference_steps}_{guidance_scale}.png")
    
    print("Inference complete!")

