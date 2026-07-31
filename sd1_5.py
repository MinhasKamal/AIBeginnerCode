import torch
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import PNDMScheduler
from transformers import CLIPTokenizer
from transformers import CLIPTextModel
from torchvision import transforms
from PIL import Image
import numpy as np

HF_TOKEN = "hf_..."

def get_pretrained_components(
        model_id: str
        ) -> tuple[AutoencoderKL, UNet2DConditionModel, PNDMScheduler, CLIPTokenizer, CLIPTextModel]:
    print(f"Loading {model_id}...")

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.float32  # or torch.float16 if running low on VRAM
    )
    print(list(pipeline.config.keys()))

    return pipeline.vae, pipeline.unet, pipeline.scheduler, pipeline.tokenizer, pipeline.text_encoder
    

# def get_pretrained_vae_and_unet(model_id: str) -> tuple[AutoencoderKL, UNet2DConditionModel]:
#     print(f"Loading {model_id}...")
#     vae = AutoencoderKL.from_pretrained(
#         model_id, 
#         subfolder="vae",
#         token=HF_TOKEN # remove this line if logging in with `hf auth login`
#     )

#     unet = UNet2DConditionModel.from_pretrained(
#         model_id, 
#         subfolder="unet",
#         token=HF_TOKEN # remove this line if logging in with `hf auth login`
#     )
    
#     return vae, unet

    
def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

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
        embedding = text_encoder(tokens)[0]
        # embedding = text_encoder(tokens).last_hidden_state

    # Free up VRAM by deleting the text encoder once the embedding is cached
    # del text_encoder, tokenizer 
    # torch.cuda.empty_cache()
    
    return embedding

    
def encode_image_to_latents(
        vae: AutoencoderKL,
        image: Image
        ) -> torch.Tensor:
    # Disable dropout layers & freezes batch normalization
    vae.eval()

    # The VAE requires images to be a multiple of 8 in dimensions.
    # We also need to normalize the image pixel values from [0, 1] to [-1, 1].
    transform = transforms.Compose([
        transforms.Resize((512, 512)), # Resize to specific dimensions (must be multiple of 8)
        transforms.ToTensor(),         # Changes from (H, W, Ch) to (Ch, H, W) & scales to [0.0, 1.0]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalizes to [-1.0, 1.0]
    ])

    # Apply transforms and add a batch dimension: [Batch_Size, Channel, Height, Width]
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 3. Encode Image into Latent Space
    print("Encoding image...")
    with torch.no_grad(): # Deactivates PyTorch's autograd engine, reducing unnecessary memory usage
        # Pass the tensor through the encoder to get the DiagonalGaussianDistribution
        latent_dist = vae.encode(image_tensor).latent_dist
        
        # For deterministic encoding, we take the mode. 
        # Alternatively, you could use latent_dist.sample() for stochastic encoding.
        # latents = latent_dist.sample()
        latents = latent_dist.mode()
        
        # Stable Diffusion requires latents to be scaled by a specific factor
        scaling_factor = vae.config.scaling_factor
        latents = latents * scaling_factor

    print(f"Original image shape: {image_tensor.shape}")
    print(f"VAE output latents shape: {latents.shape}")

    return latents


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


def run_pretrained_unet(
        unet: UNet2DConditionModel,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_encoding: torch.Tensor
        ) -> torch.Tensor:
    # Disable dropout layers & freezes batch normalization
    unet.eval()

    print("Running forward pass through the U-Net...")
    with torch.no_grad(): # Deactivates PyTorch's autograd engine, reducing unnecessary memory usage
        # The U-Net outputs a named tuple. We extract the 'sample' attribute.
        noise_latents = unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=text_encoding
        ).sample
        
    # The output is a 4-channel tensor representing the predicted noise
    print(f"U-Net output latents shape: {output_latents.shape}") 
    
    return noise_latents
    

def generate_image_from_scratch(
            unet: UNet2DConditionModel,
            vae: AutoencoderKL, 
            scheduler: PNDMScheduler, 
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            prompt: str,
            num_inference_steps: int = 40,
            guidance_scale: float = 7.5,
            generator = None
        ) -> Image:
    # Initialize the scheduler to define the timestep schedule
    scheduler.set_timesteps(num_inference_steps, device=unet.device)
    
    # need an empty encoding for unconditional image creation for classifier free guidance
    cond_embeddings = get_text_embedding(tokenizer, text_encoder, prompt)
    uncond_embeddings = get_text_embedding(tokenizer, text_encoder, "")
    # uncond_embeddings = torch.zeros_like(cond_embeddings)
    text_embedding = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
    
    # Generate initial pure Gaussian noise latents
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64), #[Batch Size, Channels, Height, Width]
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
                encoder_hidden_states=text_embedding
            ).sample

        # Apply Classifier-Free Guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_{t-1} using the scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 4. Scale and decode the fully denoised latents back to pixels
    decoded_img = decode_latents_to_image(vae, latents)
    
    print("Generation complete!")
    return decoded_img
    
def forward_pass_through_pretrained_vae_and_unet(
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        image: Image
        ) -> Image:
    img_latents = encode_image_to_latents(vae, image)
    # print(latent_representation)
    print("Latent encoding successful!")
    # latents = latents.to(device)

    # As a Stable Diffusion derivative, the U-Net structurally requires a timestep and text embeddings.
    # Dummy timestep (e.g., step 500 out of 1000 in the noise schedule)
    timestep = torch.tensor(data=[500], device=unet.device)
    # We are using empty text conditioning.
    # So, we create a tensor of zeros matching the required cross-attention dimension (e.g., 1024 or 768).
    cross_attention_dim = unet.config.cross_attention_dim
    # 1 - batch size
    # 77 - The standard maximum token length (context length) used in models like CLIP
    text_encoding_shape = (1, 77, cross_attention_dim)
    text_encoding = torch.zeros(text_encoding_shape, device=unet.device)

    noise_latents = run_pretrained_unet(unet, img_latents, timestep, text_encoding)
    
    decoded_img = decode_latents_to_image(vae, noise_latents)
    print("Latent decoding successful!")
    return decoded_img

if __name__ == "__main__":
    print("Started...")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    device = get_device()

    # prompt = "A smiling otter wearing glasses."
    prompt = "A toad fighting with a rabbit."
    out_image_path = "sd1_5_generated_img.png"
    # in_image_path = "img.jpg"
    
    vae, unet, scheduler, tokenizer, text_encoder = get_pretrained_components(model_id)
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)
    
    # image = Image.open(in_image_path).convert("RGB")
    # decoded_img = forward_pass_through_pretrained_vae_and_unet(vae, unet, image)
    decoded_img = generate_image_from_scratch(unet, vae, scheduler, tokenizer, text_encoder, prompt)
    
    decoded_img.save(out_image_path)
    print(f"Successfully saved decoded image to: {out_image_path}")

    # vae.save_pretrained("vae_sd1-5_AutoencoderKL")
    # unet.save_pretrained("unet_sd1-5_UNet2DConditionModel")
    # scheduler.save_pretrained("scheduler_sd1-5_PNDMScheduler")
    # tokenizer.save_pretrained("tokenizer_sd1-5_CLIPTokenizer")
    # text_encoder.save_pretrained("text_encoder_sd1-5_CLIPTextModel")
    # print("Individual models saved")
