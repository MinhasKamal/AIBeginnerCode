import torch
from diffusers import UNet2DConditionModel


# Hardcoding is generally discouraged, but it works perfectly for testing:
HF_TOKEN = "hf_..."

def run_untrained_unet(rgb_latents, model_id):
    """
    Initializes an untrained U-Net using Marigold's architecture and runs a forward pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb_latents = rgb_latents.to(device)
    batch_size, _, height, width = rgb_latents.shape

    # 1. Initialize the Untrained U-Net
    # We load ONLY the configuration (the blueprint), leaving the weights randomized.
    print(f"Fetching U-Net config from {model_id}...")
    config = UNet2DConditionModel.load_config(
        model_id,
        subfolder="unet",
        token=HF_TOKEN # remove this line if logging in with `hf auth login`
    )
    print(config)
    
    print("Initializing untrained U-Net from config...")
    unet = UNet2DConditionModel.from_config(config).to(device)
    
    # Ensure the model is in evaluation mode (turns off dropout, etc.)
    unet.eval()

    # 2. Prepare the 8-Channel Input
    # Marigold concatenates the RGB condition (4 channels) with the diffusion noise (4 channels)
    print("Preparing 8-channel concatenated input...")
    
    # Generate random noise simulating the starting point of the depth map generation
    noisy_depth_latents = torch.randn_like(rgb_latents).to(device)
    
    # Concatenate along the channel dimension (dim=1)
    # Shape becomes: [batch_size, 8, height, width]
    unet_input = torch.cat([rgb_latents, noisy_depth_latents], dim=1) 
    print(f"U-Net input shape:  {unet_input.shape}") 

    # 3. Prepare Required Diffusion Conditioning
    # As a Stable Diffusion derivative, the U-Net structurally requires a timestep and text embeddings.
    
    # Dummy timestep (e.g., step 500 out of 1000 in the noise schedule)
    timestep = torch.tensor([500], device=device) 
    
    # Marigold primarily uses empty text conditioning for depth estimation.
    # We create a tensor of zeros matching the required cross-attention dimension (e.g., 1024 or 768).
    cross_attention_dim = unet.config.cross_attention_dim
    encoder_hidden_states = torch.zeros((batch_size, 77, cross_attention_dim), device=device)
    
    # 4. Execute the Forward Pass
    print("Running forward pass through the untrained U-Net...")
    with torch.no_grad():
        # The U-Net outputs a named tuple. We extract the 'sample' attribute.
        unet_output = unet(
            sample=unet_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
    # The output is a 4-channel tensor representing the predicted noise or depth map update
    print(f"U-Net output shape: {unet_output.shape}") 
    
    return unet_output

def run_pretrained_unet(rgb_latents, model_id):
    """
    Initializes an untrained U-Net using Marigold's architecture and runs a forward pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb_latents = rgb_latents.to(device)
    batch_size, _, height, width = rgb_latents.shape

    # 1. Load pretrained U-Net
    print("Initializing pretrained U-Net from config...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, 
        subfolder="unet",
        token=HF_TOKEN # remove this line if logging in with `hf auth login`
    ).to(device)
        
    # 2. Ensure the model is in evaluation mode (turns off dropout, etc.)
    unet.eval()


    # 3. Prepare Required Diffusion Conditioning
    # As a Stable Diffusion derivative, the U-Net structurally requires a timestep and text embeddings.
    
    # Dummy timestep (e.g., step 500 out of 1000 in the noise schedule)
    timestep = torch.tensor([500], device=device) 
    
    # Marigold primarily uses empty text conditioning for depth estimation.
    # We create a tensor of zeros matching the required cross-attention dimension (e.g., 1024 or 768).
    cross_attention_dim = unet.config.cross_attention_dim
    encoder_hidden_states = torch.zeros((batch_size, 77, cross_attention_dim), device=device)
    
    # 4. Execute the Forward Pass
    print("Running forward pass through the untrained U-Net...")
    with torch.no_grad():
        # The U-Net outputs a named tuple. We extract the 'sample' attribute.
        unet_output = unet(
            sample=rgb_latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
    # The output is a 4-channel tensor representing the predicted noise or depth map update
    print(f"U-Net output shape: {unet_output.shape}") 
    
    return unet_output

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy 4-channel latent tensor simulating the output from the VAE script
    # Shape: [Batch Size, Channels, Height, Width]
    dummy_vae_latents = torch.randn((1, 4, 64, 64))
    
    predicted_encoding = run_untrained_unet(
        dummy_vae_latents,
        "prs-eth/marigold-v1-0" # Input Latent space shape: torch.Size([1, 8, 64, 64])
    )

    # predicted_encoding = run_pretrained_unet(
    #     dummy_vae_latents,
    #     "runwayml/stable-diffusion-v1-5" # Input Latent space shape: torch.Size([1, 4, 64, 64])
    # )
    print(predicted_encoding)
    print("Forward pass successful!")