import torch
import numpy as np


EPOCHS = 10
BATCH_SIZE = 4 # for 4 GPUs, effective batch size becomes BATCH_SIZE * 4
GRADIENT_ACCUMULATION_STEPS = 2 # effective batch size = BATCH_SIZE * GPUs * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-2

SD_XL1_IMAGE_SIZE = 1024
IMG_AUG_CROP_LEN = SD_XL1_IMAGE_SIZE // 10
IMG_AUG_NOISE_STD = 0.02
CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE = 0.2

OUT_MODEL_PATH = "depth_sd_xl1_5"


####--**--####


import matplotlib.pyplot as plt


def plot_tensor_list(
            tensor_list: list,
            file_name: str
        ):
    np_arr_list = [t.detach().cpu().permute(1, 2, 0).numpy() for t in tensor_list]
    np_arr_list = [(a+1)/2 for a in np_arr_list]
    plot_np_arr_list(np_arr_list, file_name)
    return
    

def plot_np_arr_list(
            np_arr_list: list,
            file_name: str
        ):
    fig, axes = plt.subplots(1, len(np_arr_list), figsize=(6 * len(np_arr_list), 10))

    if len(np_arr_list) > 1:
        for index, np_arr in enumerate(np_arr_list):
            axes[index].imshow(np_arr)
            # axes[index].imshow(np_arr, cmap="gray")
    else :
        axes.imshow(np_arr_list[0])
        # axes.imshow(np_arr_list[0], cmap="gray")
    
    plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.clf()
    print(f"Plot is saved in {file_name}")
    
    return


####--**--####


import os
import glob
import random
import h5py


HYPERSIM_DATASET_ROOT = "/workspace/minhas/dataset/hypersim/unzips/"
HYPERSIM_DATA_DOWN_SAMPLE_COUNT = -1 ## TEST ONLY!!!
def list_rgb_depth_in_hypersim(
        ) -> list:
    print("# Listing rgb-depth in Hypersim...")
    grouped_filepaths = []

    # Get all scene directories (e.g., ai_001_001, ai_002_001)
    scene_dirs_pattern = os.path.join(HYPERSIM_DATASET_ROOT, "ai_*")
    scene_dirs = glob.glob(scene_dirs_pattern)
    
    for scene_dir in sorted(scene_dirs):
        scene_dir = os.path.join(scene_dir, "images")
        if not os.path.exists(scene_dir):
            print(f"{scene_dir} does not exist!")
            continue
            
        # Find all final preview camera directories in this scene
        cam_preview_dirs_pattern = os.path.join(scene_dir, "scene_cam_*_final_hdf5")
        cam_preview_dirs = glob.glob(cam_preview_dirs_pattern)

        for rgb_cam_dir in cam_preview_dirs:
            # Extract the camera identifier (e.g., 'cam_00') to find its geometry match
            # depth_cam_dir = rgb_cam_dir[:-len("final_preview")] + "geometry_preview"
            depth_cam_dir = rgb_cam_dir[:-len("final_hdf5")] + "geometry_hdf5"
            
            if not os.path.exists(depth_cam_dir):
                print(f"{depth_cam_dir} does not exist!!")
                continue
                
            # Find all RGB frames inside this camera folder
            rgb_files_pattern = os.path.join(rgb_cam_dir, "frame.*.color.hdf5")
            rgb_files = glob.glob(rgb_files_pattern)
            
            for rgb_path in rgb_files:
                # Extract the exact frame number (e.g., '0000', '0001') from the filename
                filename = os.path.basename(rgb_path)
                frame_idx = filename.split(".")[1]
                
                # Construct the expected matching depth map path
                # depth_path = os.path.join(depth_cam_dir, f"frame.{frame_idx}.depth_meters.png")
                depth_path = os.path.join(depth_cam_dir, f"frame.{frame_idx}.depth_meters.hdf5")                
                # Verify that the depth map physically exists before adding the pair
                if not os.path.exists(depth_path):
                    print(f"{depth_path} does not exist!!!")
                    continue
                    
                grouped_filepaths.append((rgb_path, depth_path))
                # break ## TEST ONLY!!!

    if HYPERSIM_DATA_DOWN_SAMPLE_COUNT > 0: ## TEST ONLY!!!
        grouped_filepaths = random.sample(grouped_filepaths, HYPERSIM_DATA_DOWN_SAMPLE_COUNT)

    print(f"Total sample count: {len(grouped_filepaths)}")
    return grouped_filepaths


def read_rgb_in_hypersim(
            rgb_path: str
        ) -> np.ndarray:
    with h5py.File(rgb_path, 'r') as f:
        rgb_hdr = np.array(f["dataset"], dtype=np.float32)

    # print(f"rgb NAN count: {np.isnan(rgb_hdr).sum()}")
    rgb_hdr = np.nan_to_num( # missing/infinity instance is handled
        rgb_hdr, 
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # randomly set an exposure value
    exposure_value = random.uniform(-1.0, 1.0)

    # brightness adjustment: I = I * 2^(EV)
    # +1 exposure value doubles the light, -1 exposure value halves the light
    rgb_hdr = rgb_hdr * (2.0 ** exposure_value)

    # Standard sRGB Gamma Correction: Linear (High Dynamic Range) -> Low Dynamic Range
    # The function (Opto-Electronic Transfer Function) is the official linear-to-sRGB
    # (standard RGB) conversion standardized by Hewlett-Packard and Microsoft.
    # x <= 0.0031308 ? 12.92 * x : 1.055 * (x ** (1 / 2.4)) - 0.055
    rgb_ldr = np.where(
        rgb_hdr <= 0.0031308,
        12.92 * rgb_hdr,
        1.055 * np.power(rgb_hdr, 1.0 / 2.4) - 0.055
    )
    rgb_8bit = np.clip(rgb_ldr * 255.0, 0, 255).astype(np.uint8)

    rgb_8bit = np.transpose(rgb_8bit, (2, 0, 1)) # (H, W, Ch) -> (Ch, H, W)

    # print(f"rgb: {rgb_8bit.shape}, min: {rgb_8bit.min()}, max: {rgb_8bit.max()}")
    return rgb_8bit


HYPERSIM_MAX_DEPTH_METER = 1_000.0
def read_depth_in_hypersim(
            depth_map_path: str
        ) -> np.ndarray:
    with h5py.File(depth_map_path, 'r') as f:
        depth_map_meter = np.array(f["dataset"], dtype=np.float32)

    # print(np.isnan(depth).sum())
    depth_map_meter = np.nan_to_num( # missing/infinity depth is handled
        depth_map_meter,
        nan=HYPERSIM_MAX_DEPTH_METER,
        posinf=HYPERSIM_MAX_DEPTH_METER,
        neginf=0.0
    )
    # bound outliers: lower than 0.0 are elevated to 0.0, and values exceeding
    # HYPERSIM_MAX_DEPTH_METER are truncated to that maximum
    depth_map_meter = np.clip(depth_map_meter, a_min=0.0, a_max=HYPERSIM_MAX_DEPTH_METER)
    
    # print(f"depth: {depth_map_meter.shape}, min: {depth_map_meter.min()}, max: {depth_map_meter.max()}")
    return depth_map_meter


####--**--####


import os
import glob
import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


VKITTI_DATASET_ROOT = "/workspace/minhas/dataset/virtual_kitti/"
VKITTI_DATA_DOWN_SAMPLE_COUNT = -1 ## TEST ONLY!!!
def list_rgb_depth_in_vkitti(
        ) -> list:
    
    print("# Listing rgb-depth in Virtual KITTI...")
    grouped_filepaths = []
    
    # Get all scene directories (e.g., Scene01, Scene02, Scene06, etc.)
    scene_dirs_pattern = os.path.join(VKITTI_DATASET_ROOT, "Scene*")
    scene_dirs = glob.glob(scene_dirs_pattern)
    
    for scene_dir in sorted(scene_dirs):
        if not os.path.isdir(scene_dir):
            continue
            
        # vKITTI2 scenes have variation subfolders (e.g., 'clone', 'fog', 'morning', '15-deg-left')
        variation_dirs = glob.glob(os.path.join(scene_dir, "*"))
        
        for variation_dir in sorted(variation_dirs):
            if not os.path.isdir(variation_dir):
                continue
                
            # Paths to rgb and depth frame directories
            rgb_base_dir = os.path.join(variation_dir, "frames", "rgb")
            depth_base_dir = os.path.join(variation_dir, "frames", "depth")
            
            if not os.path.exists(rgb_base_dir) or not os.path.exists(depth_base_dir):
                continue
                
            # Iterate through cameras (e.g., Camera_0, Camera_1)
            camera_dirs = glob.glob(os.path.join(rgb_base_dir, "Camera_*"))
            
            for rgb_cam_dir in sorted(camera_dirs):
                cam_name = os.path.basename(rgb_cam_dir) # e.g., 'Camera_0'
                depth_cam_dir = os.path.join(depth_base_dir, cam_name)
                
                if not os.path.exists(depth_cam_dir):
                    print(f"{depth_cam_dir} does not exist!!")
                    continue
                    
                # Find all RGB frames inside this camera folder (vKITTI2 RGBs are usually .jpg)
                rgb_files_pattern = os.path.join(rgb_cam_dir, "rgb_*.jpg")
                rgb_files = glob.glob(rgb_files_pattern)
                
                # Fallback in case your specific download uses .png for RGB
                # if not rgb_files:
                #     rgb_files_pattern = os.path.join(rgb_cam_dir, "rgb_*.png")
                #     rgb_files = glob.glob(rgb_files_pattern)
                
                for rgb_path in rgb_files:
                    # Extract the exact frame number (e.g., '00000') from the filename 'rgb_00000.jpg'
                    filename = os.path.basename(rgb_path)
                    frame_idx = filename.split("_")[1].split(".")[0]
                    
                    # Construct the expected matching depth map path (vKITTI2 depths are 16-bit .png)
                    depth_path = os.path.join(depth_cam_dir, f"depth_{frame_idx}.png")
                    
                    # Verify that the depth map physically exists before adding the pair
                    if not os.path.exists(depth_path):
                        print(f"{depth_path} does not exist!!!")
                        continue
                        
                    grouped_filepaths.append((rgb_path, depth_path))
                    # break ## TEST ONLY!!!

    if VKITTI_DATA_DOWN_SAMPLE_COUNT > 0: ## TEST ONLY!!!
        grouped_filepaths = random.sample(grouped_filepaths, VKITTI_DATA_DOWN_SAMPLE_COUNT)
        
    print(f"Total sample count: {len(grouped_filepaths)}")
    return grouped_filepaths


def read_rgb_in_vkitti(
        rgb_path: str
    ) -> np.ndarray:
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_8bit = np.array(rgb_img, dtype=np.uint8)
    rgb_8bit = np.transpose(rgb_8bit, (2, 0, 1)) # (H, W, Ch) -> (Ch, H, W)
    return rgb_8bit


VKITTI_MAX_DEPTH_METERS: float = 1_000.0
def read_depth_in_vkitti(
            depth_path: str
        ) -> np.ndarray:

    depth_map_centimeter = np.array(Image.open(depth_path), dtype=np.float32)
    depth_map_meter = depth_map_centimeter / 100.0
    depth_map_meter = np.expand_dims(depth_map_meter, axis=0) # [H, W] -> [1, H, W]

    # Map 0.0 (sky or invalid pixels) to max depth
    depth_map_meter[depth_map_meter == 0.0] = VKITTI_MAX_DEPTH_METERS

    depth_map_meter = np.nan_to_num( # missing or infinity depth is handled
        depth_map_meter,
        nan=VKITTI_MAX_DEPTH_METERS,
        posinf=VKITTI_MAX_DEPTH_METERS,
        neginf=0.0
    )

    # bound outliers: lower than 0.0 are elevated to 0.0, and values exceeding
    # MAX_DEPTH_METERS are truncated to that maximum
    depth_map_meter = np.clip(depth_map_meter, 0.0, VKITTI_MAX_DEPTH_METERS)

    # print(depth_map_meter.shape)
    return depth_map_meter


####--**--####


def scale_depth_map_meter(
            depth_map_meter: np.ndarray
        ) -> np.ndarray:
    # depth_map_scaled = 1.0 - 2 * np.exp(-0.05 * depth_map_meter)
    depth_map_scaled = 1.0 - 2 * np.exp(-0.05 * np.power(depth_map_meter, 0.9))

    return depth_map_scaled


def unscale_depth_map_meter(
            depth_map_scaled: np.ndarray
        ) -> np.ndarray:
    # depth_map_scaled = np.clip(depth_map_scaled, -0.999999, 0.999999)
    inside_log = (1.0 - depth_map_scaled) / 2.0
    depth_map_meter = np.power(-20.0 * np.log(inside_log), 1.0 / 0.9)

    return depth_map_meter


####--**--####


from torch.utils.data import Dataset
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections.abc import Callable


class RGBDepthDataset(Dataset):
    def __init__(
                self, 
                rgb_depth_filepaths: list,
                read_rgb: Callable[[str], np.ndarray],
                read_depth: Callable[[str], np.ndarray],
                image_size: int,
                crop_len: int,
                noise_std: float
            ) -> None:
        self.rgb_depth_filepaths = rgb_depth_filepaths
        self.read_rgb = read_rgb
        self.read_depth = read_depth
        self.image_size = image_size
        self.crop_len = crop_len
        self.noise_std = noise_std
        return

    def __len__(
                self
            ) -> int:
        return len(self.rgb_depth_filepaths)

    def __getitem__(
                self,
                idx: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_path, depth_path = self.rgb_depth_filepaths[idx]
        
        rgb_8bit = self.read_rgb(rgb_path)
        rgb_tensor = torch.from_numpy(rgb_8bit.astype(np.float32))
        rgb_tensor = ((rgb_tensor / 255.0) * 2.0) - 1.0 # scale to [-1.0, 1.0]
        
        depth_map_meter = self.read_depth(depth_path)
        depth_map_scaled = scale_depth_map_meter(depth_map_meter)
        depth_tensor = torch.from_numpy(depth_map_scaled.astype(np.float32))
        depth_tensor = depth_tensor.repeat(3, 1, 1) # 3 channels for the VAE: [1, H, W] -> [3, H, W]

        target_size = (self.image_size + self.crop_len, self.image_size + self.crop_len)
        # # F.interpolate expects a batch dimension [Batch, Channel, Height, Width]
        # # So we unsqueeze(0) to fake a batch of 1, interpolate, and squeeze(0) to remove it
        rgb_tensor = F.interpolate(
            rgb_tensor.unsqueeze(0), 
            size=target_size,
            mode='nearest',
            # mode='bilinear', 
            # align_corners=False
        ).squeeze(0)
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0), 
            size=target_size,
            mode='nearest',
            # mode='bilinear', 
            # align_corners=False
        ).squeeze(0)

        # Random crop out
        top = random.randint(0, self.crop_len)
        left = random.randint(0, self.crop_len)
        rgb_tensor = TF.crop(rgb_tensor, top, left, self.image_size, self.image_size)
        depth_tensor = TF.crop(depth_tensor, top, left, self.image_size, self.image_size)

        # Apply random hflip
        if random.random() < 0.5:
            rgb_tensor = TF.hflip(rgb_tensor)
            depth_tensor = TF.hflip(depth_tensor)

        # Add subtle Gaussian noise to RGB sampled from N(0, noise_std)
        noise = torch.randn_like(rgb_tensor) * self.noise_std
        rgb_tensor = rgb_tensor + noise
        # Adding noise to a pixel value of 1.0 can push it to 1.02
        rgb_tensor = torch.clamp(rgb_tensor, -1.0, 1.0)

        # Provide blank RGB image for context-free guidance
        if random.random() < CONDITIONAL_DROPOUT_FOR_CLASSIFIER_FREE_GUIDANCE:
            rgb_tensor = torch.zeros_like(rgb_tensor)
        
        return rgb_tensor, depth_tensor


####--**--####


import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTokenizer
from transformers import CLIPPreTrainedModel
from transformers import CLIPTextModel
from transformers import CLIPTextModelWithProjection


# def get_devices(
#         ) -> list[str]:
#     if not torch.cuda.is_available():
#         print("GPU not available, using CPU!")
#         return ["cpu"]
    
#     num_gpus = torch.cuda.device_count()
#     devices = [f"cuda:{i}" for i in range(num_gpus)]
    
#     print(f"Found {num_gpus} CUDA device(s)")
#     return devices


def multiply_unet_input_capacity(
            unet: UNet2DConditionModel,
            times: int,
        ) -> UNet2DConditionModel:
    print(f"Multiplying UNet input capacity by {times}")
    with torch.no_grad():
        old_conv_in = unet.conv_in
        old_in_ch = old_conv_in.in_channels
        
        new_conv_in = nn.Conv2d(
            in_channels=times*old_in_ch, # channels
            out_channels=old_conv_in.out_channels,
            kernel_size=old_conv_in.kernel_size,
            stride=old_conv_in.stride,
            padding=old_conv_in.padding
        )

        # copy pretrained weights on both condition and noise
        weight_ratio = 1.0 / times
        for i in range(times):
            start_ch = i * old_in_ch
            end_ch = (i + 1) * old_in_ch
            new_conv_in.weight[:, start_ch:end_ch, :, :] = old_conv_in.weight.clone() * weight_ratio

        new_conv_in.bias = nn.Parameter(old_conv_in.bias.clone())
        
        unet.conv_in = new_conv_in
        unet.register_to_config(in_channels=times*old_in_ch)
        
    return unet


def get_pretrained_components(
        ) -> tuple[AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler,
        CLIPTokenizer, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection]:
    print(f"# Loading local models...")

    vae = AutoencoderKL.from_pretrained("vae_sd_xl1_AutoencoderKL")
    unet = UNet2DConditionModel.from_pretrained("unet_sd_xl1_UNet2DConditionModel")
    unet = multiply_unet_input_capacity(unet, 2)
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


from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import DDPMScheduler
from accelerate import Accelerator
# from accelerate.utils import InitProcessGroupKwargs


PLT_DATA_SKIP = 100
def produceLogAndGraph(
            loss_log: list,
            model_path : str
        ) -> None:
    os.makedirs(model_path, exist_ok=True)
    
    with open(f"{model_path}/log.json", 'w+') as f:
        json.dump(loss_log, f, indent=4)
    
    plt.rcParams["font.family"] = "Serif"
    plt.figure(figsize=(16, 6))

    plt.title("Training MSE")
    plt.xlabel(f"Iterations (Every {PLT_DATA_SKIP}th)")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.plot(loss_log['train_mse'][::PLT_DATA_SKIP], label='Train MSE')
    # plt.plot(loss_log['epoch'], loss_log['test_ssim'], label='Test MSE')
    plt.legend()

    plt.savefig(f"{model_path}/graph.pdf", format="pdf", bbox_inches="tight")
    print(f"# graph saved in {model_path}")
    # plt.show()
    plt.clf()


DATA_LOADER_WORKERS = 2 # Adjust based on CPU cores
def run_trainer(
            dataset: Dataset,
            orig_img_h: int,
            orig_img_w: int,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            weight_decay: float,
            out_model_path: str,
            accelerator: Accelerator
        ) -> None:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=True
    )
    
    device = accelerator.device

    vae, unet, scheduler, tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2 = get_pretrained_components()
    
    vae = vae.to(device)
    vae.requires_grad_(False) # Freeze model weights and never compute gradients for them
    vae.eval() # Layer behavior adjustment, like- disable dropout layers & freeze batch normalization
    
    unet = unet.to(device)
    unet.requires_grad_(True)
    unet.enable_gradient_checkpointing() # reduces some memory usage but increases time
    
    scheduler = DDPMScheduler.from_config(scheduler.config)
    
    text_encoder_1 = text_encoder_1.to(device)
    text_encoder_1.requires_grad_(False)
    text_encoder_1.eval()
    
    text_encoder_2 = text_encoder_2.to(device)
    text_encoder_2.requires_grad_(False)
    text_encoder_2.eval()

    text_embedding, text_embedding_pooled = get_text_embedding_2("", tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2)
    
    # SDXL was trained on random center crops, causing its UNet to learn that
    # objects are often sliced off. By passing (0, 0) during inference to force
    # the UNet to generate uncropped compositions.
    # SDXL included lower-resolution images scaled up. By conditioning on original 
    # size, the model learns the difference between upscaled blurriness and native
    # high-frequency detail.
    time_id = torch.tensor(
        # [[SD_XL1_IMAGE_SIZE, SD_XL1_IMAGE_SIZE, 0, 0, SD_XL1_IMAGE_SIZE, SD_XL1_IMAGE_SIZE]],
        [[orig_img_h, orig_img_w, 0, 0, SD_XL1_IMAGE_SIZE, SD_XL1_IMAGE_SIZE]],
        # orig height, orig width, crop top, crop left, trgt height, trgt width
        # for hypersim dataset- 768, 1024, 0, 0, 1024, 1024
        device=device,
        dtype=text_embedding.dtype
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if accelerator.is_main_process:
        loss_log = {
            'epoch': [],
            'train_mse': [],
        }

        print("# Training...")

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    for epoch in range(epochs):
        unet.train()

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=not accelerator.is_main_process # only show progress bar on the main process
        )

        for rgb_imgs, depth_imgs in progress_bar:
            with accelerator.accumulate(unet):

                sub_batch_size = rgb_imgs.shape[0]

                rgb_latent_batch = encode_image_batch_to_latents(vae, rgb_imgs)
                depth_latent_batch = encode_image_batch_to_latents(vae, depth_imgs)

                # Sample random timestep for each image in the batch
                timestep_batch = torch.randint(
                    low=0,
                    high=scheduler.config.num_train_timesteps, 
                    size=(sub_batch_size,),
                    device=device
                ).long()
                # Sample Gaussian noise
                noise = torch.randn_like(depth_latent_batch)
                # The timestep_batch dictates how much of that noise is blended into the clean latents
                noisy_depth_latent_batch = scheduler.add_noise(depth_latent_batch, noise, timestep_batch)

                input_latent = torch.cat([rgb_latent_batch, noisy_depth_latent_batch], dim=1)

                text_embedding_batch = text_embedding.repeat(sub_batch_size, 1, 1)
                text_embedding_pooled_batch = text_embedding_pooled.repeat(sub_batch_size, 1)
                time_id_batch = time_id.repeat(sub_batch_size, 1)
                added_cond_kwargs = {
                    "text_embeds": text_embedding_pooled_batch,
                    "time_ids": time_id_batch
                }

                noise_pred = unet(
                    sample=input_latent,
                    timestep=timestep_batch,
                    encoder_hidden_states=text_embedding_batch,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # Though we added scaled noise (depending on timesteps) to the image, we calculate
                # the loss against the original, unscaled noise. Because if we tried to predict the
                # scaled noise, the target values would be huge at step 999 and tiny at step 1. Deep
                # neural networks struggle to learn this. Whether the U-Net is looking at heavy noise
                # or light noise, the statistical properties of its target remain exactly the same.
                # This makes the loss gradient smooth and predictable.
                loss = F.mse_loss(noise_pred, noise)

                # loss.backward()
                accelerator.backward(loss)
                # (Optional but highly recommended) Clip gradients to prevent random spikes
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.step()

                # Clear the old gradients before calculating new ones
                optimizer.zero_grad()
                # optimizer.zero_grad(set_to_none=True)  # may reduce some memory usage
    
                if accelerator.is_main_process:
                    loss_log['epoch'].append(epoch+1)
                    loss_log['train_mse'].append(loss.item())
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(out_model_path)
            produceLogAndGraph(loss_log, out_model_path)
            print(f"Successfully saved model to: {out_model_path}")

    return

    

if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    accelerator.print("# Started!")
    
    # rgb_depth_filepaths = list_rgb_depth_in_hypersim()
    # dataset = RGBDepthDataset(rgb_depth_filepaths, read_rgb_in_hypersim, 
    #                           read_depth_in_hypersim, SD_XL1_IMAGE_SIZE, IMG_AUG_CROP_LEN, IMG_AUG_NOISE_STD)
    # orig_img_h, orig_img_w = 768, 1024

    rgb_depth_filepaths = list_rgb_depth_in_vkitti()
    dataset = RGBDepthDataset(rgb_depth_filepaths, read_rgb_in_vkitti, 
                              read_depth_in_vkitti, SD_XL1_IMAGE_SIZE, IMG_AUG_CROP_LEN, IMG_AUG_NOISE_STD)
    orig_img_h, orig_img_w = 375, 1242
    
    if accelerator.is_main_process: plot_tensor_list(dataset.__getitem__(2), "sd_xl1_depth_trainer_test")
    run_trainer(dataset, orig_img_h, orig_img_w, EPOCHS, BATCH_SIZE,
                LEARNING_RATE, WEIGHT_DECAY, OUT_MODEL_PATH, accelerator)

    accelerator.end_training()
    accelerator.print("# Training finished!")


"""
accelerate config
-------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower[yes/NO]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:4
What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:
----------------------------------------------------------------------------------------------------------

accelerate launch sd_xl1_depth_trainer.py

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision no \
    sd_xl1_depth_trainer.py

ctrl+z -> kill -9 %1
"""
