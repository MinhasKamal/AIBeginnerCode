import torch
from diffusers import Flux2Pipeline
# from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxPipeline
# from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer

# Hardcoding is generally discouraged, but it works perfectly for testing:
HF_TOKEN = "hf_..."

# 1. Define the SOTA heavyweight model
print(f"1. Define the SOTA heavyweight model")
model_id = "black-forest-labs/FLUX.2-dev" # 32-Billion parameter model at uncompressed 16-bit precision
# (If you prefer the 12B parameter version, use "black-forest-labs/FLUX.1-dev")

# Need to distribute the model parts to two 80GB GPUs 
# my_device_map = {
#     # Put the massive 32B main transformer onto GPU 0 (~64GB)
#     "transformer": "cuda:0",
    
#     # Push everything else onto GPU 1
#     "text_encoder": "cuda:1",
#     "text_encoder_2": "cuda:1",
#     "tokenizer": "cuda:1",
#     "tokenizer_2": "cuda:1",
#     "vae": "cuda:1",
#     "scheduler": "cuda:1"
# }

# 2. Load your core modules onto your separate GPU kernels
print(f"2. Load your core modules onto your separate GPU kernels")
# transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda:0")
# vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda:1")
# text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16).to("cuda:1")
# tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
# text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to("cuda:1")
# tokenizer_2 = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

# Load the pipeline 
# We use bfloat16 to fit the 32B parameters into ~64GB of your 80GB VRAM natively
pipe = Flux2Pipeline.from_pretrained(
    model_id, 
    # transformer=transformer,
    # vae=vae,
    # text_encoder=text_encoder,
    # tokenizer=tokenizer,
    # text_encoder_2=text_encoder_2,
    # tokenizer_2=tokenizer_2,
    # feature_extractor=None,
    # image_encoder=None,
    torch_dtype=torch.bfloat16,
    # device_map="balanced",
    # max_memory={
    #     0: "75GiB",   # GPU 0
    #     1: "75GiB",   # GPU 1
    #     "cpu": "128GiB",  # CPU RAM overflow (rarely needed for FLUX.1-dev)
    # },
    token=HF_TOKEN # remove this line if logging in with `hf auth login`
)
pipe.enable_model_cpu_offload()

# Disable the safety checker for research / private use; remove if needed.
# pipe.safety_checker = None

# 3. Move the model to the GPU
print(f"3. Move the model to the GPU")
# Because you have 80GB, you don't need CPU offloading. 
# pipe = pipe.to("cuda") # cannot load the image encoder (VAE) and
                         # text encoders (T5-XXL and CLIP ViT-L/14) and
                         # core flux transformer into 1 80GB GPU
print(f"Loading {model_id} into specific GPUs... (This will take 10 min)")

# # Put the Text Encoders on GPU 1 
# pipe.text_encoder.to("cuda:1")
# pipe.text_encoder_2.to("cuda:1")

# # Put your VAE on GPU 1 (Where your custom multi-view adapter will live)
# pipe.vae.to("cuda:1")

# # Dedicate GPU 0 to the heavy 32B Diffusion Transformer
# pipe.transformer.to("cuda:0")

# Optional: Enable memory-efficient attention if you plan to generate massive resolutions
# pipe.enable_xformers_memory_efficient_attention() 

# 4. Define your computer vision/scene prompt
print(f"4. Define your computer vision/scene prompt")
prompt = (
    "A highly detailed, photorealistic render of a complex chemistry lab. "
    "Glass beakers, a microscope, and a robotic arm are visible on a sleek workbench. "
    "Cinematic lighting, shallow depth of field, 8k resolution."
)

print("Generating image...")

# 5. Run inference
print(f"5. Run inference")
# FLUX dev models typically require 25-50 steps for pristine quality
image = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=2.5, # FLUX generally uses lower guidance scales than older SD models
    height=1024,
    width=1024
).images[0]

# 6. Save the output
print(f"6. Save the output")
output_path = "chemlab.png"
image.save(output_path)
print(f"Success! Image saved to {output_path}")
