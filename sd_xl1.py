import torch
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
from transformers import CLIPPreTrainedModel
from transformers import CLIPTextModel
from transformers import CLIPTextModelWithProjection
from torchvision import transforms
from PIL import Image
import numpy as np

HF_TOKEN = "hf_..."

def get_pretrained_components(
        model_id: str
        ) -> tuple[AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler,
        CLIPTokenizer, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection]:
    print(f"Loading {model_id}...")

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.float32  # or torch.float16 if running low on VRAM
    )
    print(list(pipeline.config.keys()))

    # print(type(pipeline.vae))
    # print(type(pipeline.unet))
    # print(type(pipeline.scheduler))
    # print(type(pipeline.tokenizer))
    # print(type(pipeline.text_encoder))
    # print(type(pipeline.tokenizer_2))
    # print(type(pipeline.text_encoder_2))

    return (pipeline.vae, pipeline.unet, pipeline.scheduler, pipeline.tokenizer, pipeline.text_encoder,
            pipeline.tokenizer_2, pipeline.text_encoder_2)


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def get_text_embedding(
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPPreTrainedModel,
        text: str
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
        tokenizer_1: CLIPTokenizer,
        text_encoder_1: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        text: str
        ) -> tuple[torch.Tensor, torch.Tensor]:

    token_embeds_1, _  = get_text_embedding(tokenizer_1, text_encoder_1, text)
    token_embeds_2, pooled_embeds_2 = get_text_embedding(tokenizer_2, text_encoder_2, text)
    text_embedding = torch.cat([token_embeds_1, token_embeds_2], dim=-1)
    # print(f"final text embed shape {text_embedding.shape}")
    
    return text_embedding, pooled_embeds_2


# def encode_image_to_latents(
#         vae: AutoencoderKL,
#         image: Image
#         ) -> torch.Tensor:
#     # Disable dropout layers & freezes batch normalization
#     vae.eval()

#     # The VAE requires images to be a multiple of 8 in dimensions.
#     # We also need to normalize the image pixel values from [0, 1] to [-1, 1].
#     transform = transforms.Compose([
#         transforms.Resize((512, 512)), # Resize to specific dimensions (must be multiple of 8)
#         transforms.ToTensor(),         # Changes from (H, W, Ch) to (Ch, H, W) & scales to [0.0, 1.0]
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalizes to [-1.0, 1.0]
#     ])

#     # Apply transforms and add a batch dimension: [Batch_Size, Channel, Height, Width]
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     # 3. Encode Image into Latent Space
#     print("Encoding image...")
#     with torch.no_grad(): # Deactivates PyTorch's autograd engine, reducing unnecessary memory usage
#         # Pass the tensor through the encoder to get the DiagonalGaussianDistribution
#         latent_dist = vae.encode(image_tensor).latent_dist
        
#         # For deterministic encoding, we take the mode. 
#         # Alternatively, you could use latent_dist.sample() for stochastic encoding.
#         # latents = latent_dist.sample()
#         latents = latent_dist.mode()
        
#         # Stable Diffusion requires latents to be scaled by a specific factor
#         scaling_factor = vae.config.scaling_factor
#         latents = latents * scaling_factor

#     print(f"Original image shape: {image_tensor.shape}")
#     print(f"VAE output latents shape: {latents.shape}")

#     return latents


def decode_latents_to_image(
        model: AutoencoderKL,
        latents: torch.Tensor
        ) -> Image:
    # Disable dropout layers & freezes batch normalization
    vae.eval() 

    # Reverse the Latent Scaling Factor
    scaling_factor = vae.config.scaling_factor
    scaled_latents = latents / scaling_factor

    print("Decoding latents through VAE...")
    with torch.no_grad():
        # vae.decode returns a DecoderOutput object; we extract the 'sample' attribute
        decoded_tensor = vae.decode(scaled_latents).sample

    # The VAE outputs values in the range [-1.0, 1.0]. We map this back to [0.0, 1.0].
    decoded_tensor = (decoded_tensor / 2.0 + 0.5).clamp(0.0, 1.0)
    
    # Remove the batch dimension [1, C, H, W] -> [C, H, W]
    decoded_tensor = decoded_tensor.squeeze(0)
    
    # Move to CPU and convert to a PIL Image
    decoded_tensor = decoded_tensor.cpu()
    to_pil = transforms.ToPILImage()
    output_image = to_pil(decoded_tensor)
    
    return output_image


# def run_pretrained_unet(
#         unet: UNet2DConditionModel,
#         latents: torch.Tensor,
#         timestep: torch.Tensor,
#         text_encoding: torch.Tensor
#         ) -> torch.Tensor:
#     # Disable dropout layers & freezes batch normalization
#     unet.eval()

#     print("Running forward pass through the U-Net...")
#     with torch.no_grad(): # Deactivates PyTorch's autograd engine, reducing unnecessary memory usage
#         # The U-Net outputs a named tuple. We extract the 'sample' attribute.
#         noise_latents = unet(
#             sample=latents,
#             timestep=timestep,
#             encoder_hidden_states=text_encoding
#         ).sample
        
#     # The output is a 4-channel tensor representing the predicted noise
#     print(f"U-Net output latents shape: {output_latents.shape}") 
    
#     return noise_latents


def generate_image_from_scratch(
            vae: AutoencoderKL, 
            unet: UNet2DConditionModel, 
            scheduler: EulerDiscreteScheduler,
            text_embedding: torch.Tensor, 
            added_cond_kwargs: dict,
            num_inference_steps: int = 40,
            guidance_scale: float = 7.5,
            generator=None
            # generator = torch.Generator(device="cpu").manual_seed(42)
        ) -> Image:
    batch_size = 1
    scheduler.set_timesteps(num_inference_steps, device=unet.device)
    
    # Generate initial pure Gaussian noise latents
    latents = torch.randn(
        (batch_size, unet.config.in_channels, 128, 128), #[Batch Size, Channels, Height, Width]
        generator=generator,
        device=unet.device,
        dtype=unet.dtype
    )
    # Scale the initial latents by the scheduler's required initial noise scale
    latents = latents * scheduler.init_noise_sigma

    # Denoising Loop
    for t in scheduler.timesteps:
        # Expand latents if doing Classifier-Free Guidance (CFG)
        # assuming text_embedding contains both [unconditional_cond, conditional_cond]
        latent_model_input = torch.cat([latents] * 2)
        
        # Scale model input based on current timestep constraints (specific to certain schedulers)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict the noise residual using U-Net
        with torch.no_grad():
            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embedding,
                added_cond_kwargs=added_cond_kwargs
            ).sample

        # Apply Classifier-Free Guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_{t-1} using the scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Scale and decode the fully denoised latents back to pixels
    decoded_img = decode_latents_to_image(vae, latents)

    print("Generation complete!")
    return decoded_img

    
# def forward_pass_through_pretrained_vae_and_unet(
#         vae: AutoencoderKL,
#         unet: UNet2DConditionModel,
#         image: Image
#         ) -> Image:
#     img_latents = encode_image_to_latents(vae, image)
#     # print(latent_representation)
#     print("Latent encoding successful!")
#     # latents = latents.to(device)

#     # As a Stable Diffusion derivative, the U-Net structurally requires a timestep and text embeddings.
#     # Dummy timestep (e.g., step 500 out of 1000 in the noise schedule)
#     timestep = torch.tensor(data=[500], device=unet.device)
#     # We are using empty text conditioning.
#     # So, we create a tensor of zeros matching the required cross-attention dimension (e.g., 1024 or 768).
#     cross_attention_dim = unet.config.cross_attention_dim
#     # 1 - batch size
#     # 77 - The standard maximum token length (context length) used in models like CLIP
#     text_encoding_shape = (1, 77, cross_attention_dim)
#     text_encoding = torch.zeros(text_encoding_shape, device=unet.device)

#     noise_latents = run_pretrained_unet(unet, img_latents, timestep, text_encoding)
    
#     decoded_img = decode_latents_to_image(vae, noise_latents)
#     print("Latent decoding successful!")
#     return decoded_img


if __name__ == "__main__":
    print("Started...")
    
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    device = get_device()
    batch_size = 1

    # prompt = "A smiling otter wearing glasses."
    prompt = "A toad fighting with a rabbit."
    out_image_path = "sd_xl1_generated_img.png"
    
    vae, unet, scheduler, tokenizer, text_encoder, tokenizer_2, text_encoder_2 = get_pretrained_components(model_id)
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)

    cond_embeddings, cond_embedding_pooled = get_text_embedding_2(
        tokenizer, text_encoder, tokenizer_2, text_encoder_2, prompt)
    uncond_embeddings, uncond_embedding_pooled = get_text_embedding_2(
        tokenizer, text_encoder, tokenizer_2, text_encoder_2, "")
    # uncond_embeddings = torch.zeros_like(cond_embeddings)
    # uncond_embedding_pooled = torch.zeros_like(cond_embedding_pooled)
    text_embedding = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

    # representing spatial attributes of input and output image
    time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]],
        device=device,
        dtype=text_embedding.dtype
    ).repeat(batch_size, 1)
    added_cond_kwargs = {
        "text_embeds": torch.cat([uncond_embedding_pooled, cond_embedding_pooled], dim=0),
        "time_ids": torch.cat([time_ids, time_ids], dim=0)
    }

    decoded_img = generate_image_from_scratch(vae, unet, scheduler, text_embedding, added_cond_kwargs)
    
    decoded_img.save(out_image_path)
    print(f"Successfully saved decoded image to: {out_image_path}")

    # vae.save_pretrained("vae_sd_xl1_AutoencoderKL")
    # unet.save_pretrained("unet_sd_xl1_UNet2DConditionModel")
    # scheduler.save_pretrained("scheduler_sd_xl1_EulerDiscreteScheduler")
    # tokenizer.save_pretrained("tokenizer_sd_xl1_CLIPTokenizer")
    # text_encoder.save_pretrained("text_encoder_sd_xl1_CLIPTextModel")
    # tokenizer_2.save_pretrained("tokenizer_2_sd_xl1_CLIPTokenizer")
    # text_encoder_2.save_pretrained("text_encoder_2_sd_xl1_CLIPTextModelWithProjection")
    # print("Individual models saved")
