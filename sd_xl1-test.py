import torch
from diffusers import StableDiffusionXLPipeline


HF_TOKEN = "hf_..."
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# prompt = "A smiling otter wearing glasses."
prompt = "A toad fighting with a rabbit."
out_image_path = "sd_xl1_generated_img.png"

print(f"Loading {model_id}...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    token=HF_TOKEN,
    torch_dtype=torch.float32,  # or torch.float16 if running low on VRAM
    # use_safetensors=True
)
print(list(pipe.config.keys()))

# if larger model compared to GPU VRAM
# pipe.enable_model_cpu_offload()
# Or, load the whole into GPU
pipe = pipe.to("cuda")

print(f"Running inference...")
image = pipe(
    prompt=prompt,
    target_size=(1024, 1024)
    # num_inference_steps=50,
    # guidance_scale=2.5,
).images[0]

image.save(out_image_path)
print(f"Success! Image saved to {out_image_path}")
