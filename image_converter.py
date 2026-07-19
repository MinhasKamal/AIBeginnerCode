from pathlib import Path
from PIL import Image

# ==========================
# Configuration
# ==========================
input_folder = "/workspace/minhas/dataset/test_depth/depth_0"
output_folder = "/workspace/minhas/dataset/test_depth/depth"

prefix = "image"      # Output filename prefix
start_index = 1       # Starting number
digits = 4            # Number of digits (0001, 0002, ...)

# Supported image formats
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ==========================
# Create output folder
# ==========================
input_path = Path(input_folder)
output_path = Path(output_folder)
output_path.mkdir(parents=True, exist_ok=True)

# ==========================
# Convert and rename
# ==========================
image_files = sorted(
    [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
)

for i, img_path in enumerate(image_files, start=start_index):
    output_name = f"{prefix}_{i:0{digits}d}.png"
    output_file = output_path / output_name

    with Image.open(img_path) as img:
        # Convert to RGB if necessary (e.g., JPEG)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        img.save(output_file, format="PNG")

print(f"Converted {len(image_files)} images.")
print(f"Saved to: {output_path}")
