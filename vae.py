import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
import numpy as np

# Hardcoding is generally discouraged, but it works perfectly for testing:
HF_TOKEN = "hf_..."

def get_pretrained_model(model_id):
    # Load the Pretrained VAE
    # We load Stable Diffusion's AutoencoderKL.
    print(f"Loading VAE from {model_id}...")
    vae = AutoencoderKL.from_pretrained(
        model_id, 
        # subfolder="vae",
        token=HF_TOKEN # remove this line if logging in with `hf auth login`
    )
    
    return vae

def encode_image_to_latents(image_path, model):
    """
    Loads a pretrained VAE and encodes an image into latent space.
    """
    # 1. Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = model.to(device)
    model.eval() # Ensure the model is in evaluation mode

    # 2. Load and Preprocess the Image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")

    # The VAE requires images to be a multiple of 8 in dimensions.
    # We also need to normalize the image pixel values from [0, 1] to [-1, 1].
    transform = transforms.Compose([
        transforms.Resize((512, 512)), # Resize to specific dimensions (must be multiple of 8)
        transforms.ToTensor(),         # Converts to [0.0, 1.0]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalizes to [-1.0, 1.0]
    ])

    # Apply transforms and add a batch dimension: [Batch_Size, Channel, Height, Width]
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 3. Encode Image into Latent Space
    print("Encoding image...")
    with torch.no_grad():
        # Pass the tensor through the encoder to get the DiagonalGaussianDistribution
        latent_dist = model.encode(image_tensor).latent_dist
        
        # For deterministic encoding, we take the mode. 
        # Alternatively, you could use latent_dist.sample() for stochastic encoding.
        # latents = latent_dist.sample()
        latents = latent_dist.mode()
        
        # Stable Diffusion requires latents to be scaled by a specific factor
        scaling_factor = model.config.scaling_factor
        latents = latents * scaling_factor

    print(f"Original image shape: {image_tensor.shape}")
    print(f"Latent space shape: {latents.shape}") # Will be [1, 4, 64, 64] if input is 512x512
    
    return latents



def decode_latents_to_image(latents, model, output_path):
    """
    Takes a latent tensor, reverses the scaling factor, decodes it via the VAE,
    denormalizes the pixel values, and saves it as a PIL image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latents = latents.to(device)

    # 1. Load the Pretrained VAE
    vae = model.to(device)
    vae.eval()

    # 2. Reverse the Latent Scaling Factor
    # This is critical! If you don't divide by this factor, the decoded image will look distorted/blurry.
    print("Unscaling latents...")
    scaling_factor = vae.config.scaling_factor
    scaled_latents = latents / scaling_factor

    # 3. Decode the Latents
    print("Decoding latents through VAE...")
    with torch.no_grad():
        # vae.decode returns a DecoderOutput object; we extract the 'sample' attribute
        decoded_tensor = vae.decode(scaled_latents).sample

    # 4. Post-Process the Decoded Image Tensor
    print("Post-processing and denormalizing...")
    # The VAE outputs values in the range [-1.0, 1.0]. We map this back to [0.0, 1.0].
    decoded_tensor = (decoded_tensor / 2.0 + 0.5).clamp(0.0, 1.0)
    
    # Remove the batch dimension [1, C, H, W] -> [C, H, W]
    decoded_tensor = decoded_tensor.squeeze(0)
    
    # Move to CPU and convert to a PIL Image
    decoded_tensor = decoded_tensor.cpu()
    to_pil = transforms.ToPILImage()
    output_image = to_pil(decoded_tensor)

    # 5. Save and Return the Image
    output_image.save(output_path)
    print(f"Successfully saved decoded image to: {output_path}")
    
    return output_image

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the path to your actual image
    sample_image_path = "img.jpg"
    
    try:
        vae = get_pretrained_model(
            "stabilityai/sdxl-vae" # Output Latent space shape: torch.Size([1, 4, 64, 64])
            # "prs-eth/marigold-v1-0" # Output Latent space shape: torch.Size([1, 4, 64, 64])
            # "black-forest-labs/FLUX.1-dev" # Output Latent space shape: torch.Size([1, 16, 64, 64])
            # "black-forest-labs/FLUX.2-dev" # Output Latent space shape: torch.Size([1, 32, 64, 64])
        )
        
        latent_representation = encode_image_to_latents(sample_image_path, vae)
        print(latent_representation)
        print("Latent encoding successful!")
        
        decoded_img = decode_latents_to_image(latent_representation, vae, "decoded_output.png")
        print("Latent decoding successful!")
    except FileNotFoundError:
        print(f"Please provide a valid image path. Could not find: {sample_image_path}")

