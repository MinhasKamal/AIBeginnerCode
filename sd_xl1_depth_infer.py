import torch
import numpy as np


UNET_PATH = "depth_sd_xl1_0"
IMAGE_NAME = "prof_home.jpg"
SD_XL1_IMAGE_SIZE = 1024
INFERENCE_STEPS_LIST = [15, 20, 25, 30, 40, 60, 100, 150, 250, 500, 990]
GUIDANCE_SCALE_LIST = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 7.0, 10.0]


####--**--####


from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
from transformers import CLIPPreTrainedModel
from transformers import CLIPTextModel
from transformers import CLIPTextModelWithProjection


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def get_pretrained_components(
        ) -> tuple[AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler,
        CLIPTokenizer, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection]:
    print(f"# Loading local models...")

    vae = AutoencoderKL.from_pretrained("vae_sd_xl1_AutoencoderKL")
    unet = UNet2DConditionModel.from_pretrained(UNET_PATH)
    scheduler = EulerDiscreteScheduler.from_pretrained("scheduler_sd_xl1_EulerDiscreteScheduler")
    tokenizer_1 = CLIPTokenizer.from_pretrained("tokenizer_1_sd_xl1_CLIPTokenizer")
    text_encoder_1 = CLIPTextModel.from_pretrained("text_encoder_1_sd_xl1_CLIPTextModel")
    tokenizer_2 = CLIPTokenizer.from_pretrained("tokenizer_2_sd_xl1_CLIPTokenizer")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("text_encoder_2_sd_xl1_CLIPTextModelWithProjection")

    return (vae, unet, scheduler, tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2)


def get_text_embedding(
        text: str,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPPreTrainedModel,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, # Always good practice to include truncation
        return_tensors="pt" # Request PyTorch tensors ("pt")
    ).input_ids.to(text_encoder.device)

    with torch.no_grad():
        output = text_encoder(tokens, output_hidden_states=True)

        token_embeds = output.last_hidden_state
        # Extract the penultimate (second-to-last) layer
        # token_embeds = output.hidden_states[-2]
        # print(f"token-by-token features {token_embeds.shape}")
        
        if hasattr(output, "text_embeds"):
            pooled_embeds = output.text_embeds
            # print(f"pooled features representing the whole {pooled_embeds.shape}")
        else:
            pooled_embeds = None
        
    # Free up VRAM by deleting the text encoder once the embedding is cached
    # del text_encoder, tokenizer 
    # torch.cuda.empty_cache()

    return token_embeds, pooled_embeds


def get_text_embedding_2(
        text: str,
        tokenizer_1: CLIPTokenizer,
        text_encoder_1: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        ) -> tuple[torch.Tensor, torch.Tensor]:

    token_embeds_1, _  = get_text_embedding(text, tokenizer_1, text_encoder_1)
    token_embeds_2, pooled_embeds_2 = get_text_embedding(text, tokenizer_2, text_encoder_2)
    text_embedding = torch.cat([token_embeds_1, token_embeds_2], dim=-1)
    # print(f"final text embed shape {text_embedding.shape}")
    
    return text_embedding, pooled_embeds_2


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
    # print(f"Latent space shape: {latents.shape}")

    return latents


####--**--####

    
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2


def get_image_tensor(
        ) -> torch.Tensor:
    ## Prepare the Input RGB Image
    print(f"Image name: {IMAGE_NAME}")
    rgb_img = Image.open(IMAGE_NAME).convert("RGB")
    rgb_tensor = TF.to_tensor(rgb_img) # (H, W, Ch) -> (Ch, H, W) & scale to [0.0, 1.0]
    rgb_tensor = (rgb_tensor * 2 - 1.0) # [0.0, 1.0] -> [-1.0, 1.0]

    target_size = (SD_XL1_IMAGE_SIZE, SD_XL1_IMAGE_SIZE)
    # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
    # # So we unsqueeze(0) to fake a batch of 1, interpolate, and squeeze(0) to remove it
    rgb_tensor = F.interpolate(
        rgb_tensor.unsqueeze(0), 
        size=target_size,
        mode='nearest',
        # mode='bilinear', 
        # align_corners=False
    ).squeeze(0)

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


def infer_classifier_free(
        rgb_tensor: torch.Tensor,
        guidance_scale: float,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: EulerDiscreteScheduler,
        text_embedding: torch.Tensor,
        added_cond_kwargs: dict,
        ) -> torch.Tensor:
    print("Running classifier free...")
    blank_tensor = torch.zeros_like(rgb_tensor)
    
    with torch.no_grad():
        # Encode RGB into latents
        rgb_latents = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor
        blank_latents = vae.encode(blank_tensor).latent_dist.mode() * vae.config.scaling_factor
        
        # Initialize pure random noise for the depth map
        depth_latents = torch.randn_like(rgb_latents)
        # Scale the initial latents by the scheduler's required initial noise scale
        # depth_latents = depth_latents * scheduler.init_noise_sigma
        
        for t in scheduler.timesteps:
            # Concatenate RGB latents and noisy depth latents along the channel dimension
            rgb_input = torch.cat([rgb_latents, depth_latents], dim=1)
            blank_input = torch.cat([blank_latents, depth_latents], dim=1)
            unet_input = torch.cat([blank_input, rgb_input], dim=0)

            # Scale model input based on current timestep constraints (specific to certain schedulers)
            unet_input = scheduler.scale_model_input(unet_input, t)
            
            # Predict the noise residual
            noise_pred = unet(
                sample=unet_input,
                timestep=t,
                encoder_hidden_states=text_embedding,
                added_cond_kwargs=added_cond_kwargs
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


####--**--####


if __name__ == "__main__":
    print("# Started!")
    batch_size = 2 # for context-free guidance
    
    device = get_device()
    
    # Load Your Saved Models
    vae, unet, scheduler, tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2 = get_pretrained_components()
    # Set models to evaluation mode
    vae = vae.to(device)
    vae.eval()
    unet = unet.to(device)
    unet.eval()
    text_encoder_1 = text_encoder_1.to(device)
    text_encoder_1.eval()
    text_encoder_2 = text_encoder_2.to(device)
    text_encoder_2.eval()
    
    # Initialize an empty text embedding
    text_embedding, text_embedding_pooled = get_text_embedding_2("", tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2)
    text_embedding_batch = text_embedding.repeat(batch_size, 1, 1)
    text_embedding_pooled_batch = text_embedding_pooled.repeat(batch_size, 1)

    # Representing spatial attributes of input and output image
    time_ids = torch.tensor(
        [[SD_XL1_IMAGE_SIZE, SD_XL1_IMAGE_SIZE, 0, 0, SD_XL1_IMAGE_SIZE, SD_XL1_IMAGE_SIZE]],
        device=device,
        dtype=text_embedding.dtype
    ).repeat(batch_size, 1)
    added_cond_kwargs = {
        "text_embeds": text_embedding_pooled_batch,
        "time_ids": time_ids
    }
        
    rgb_tensor = get_image_tensor().unsqueeze(0).to(device)
    save_image_tensor(rgb_tensor, f"{UNET_PATH}/{IMAGE_NAME}_in.png")

    for inference_steps in INFERENCE_STEPS_LIST:
        for guidance_scale in GUIDANCE_SCALE_LIST:
            scheduler.set_timesteps(inference_steps)
            depth_output = infer_classifier_free(rgb_tensor, guidance_scale, vae, unet, scheduler, text_embedding_batch, added_cond_kwargs)
            save_image_tensor_colormap(depth_output, f"{UNET_PATH}/{IMAGE_NAME}_{inference_steps}_{guidance_scale}.png")

    print("# Inference complete!")

