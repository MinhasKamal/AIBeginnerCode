import torch
from diffusers import StableDiffusionPipeline


HF_TOKEN = "hf_..."
model_id = "runwayml/stable-diffusion-v1-5"
# prompt = "A smiling otter wearing glasses."
prompt = "A toad fighting with a rabbit."
out_image_path = "sd1_5_generated_img.png"

print(f"Loading {model_id}...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    token=HF_TOKEN,
    torch_dtype=torch.float32  # or torch.float16 if running low on VRAM
)
print(list(pipe.config.keys()))

# if larger model compared to GPU VRAM
# pipe.enable_model_cpu_offload()
# Or, load the whole into GPU
pipe = pipe.to("cuda")

print(f"Running inference...")
image = pipe(
    prompt=prompt,
    # num_inference_steps=50,
    # guidance_scale=2.5,
).images[0]

image.save(out_image_path)
print(f"Success! Image saved to {out_image_path}")
