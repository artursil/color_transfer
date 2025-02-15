# %%

import numpy as np
import torch
import math
import fastai
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from dataloader import get_imagenette_dataloader
from quantize import quantize_img, plot_imgs

device = "cuda"
# %%
def method_helper(o): return list(filter(lambda x: x[0] != "_", dir(o)))
# %%
dl = get_imagenette_dataloader("/mnt/wd/datasets/imagenette2")
# %%
batch = next(iter(dl))
batch.show_batch()
# %%
print(type(batch))
# %%
# method_helper(type(batch))
# %%
img = batch.decode_batch(batch.one_batch())[0]
# %%
img = img[0].numpy().astype(np.uint8).transpose((1,2,0))
print(img.shape)
image = Image.fromarray(img)
image
# %%
num_colors = 8
quant_img = quantize_img(image, num_colors)
plot_imgs(img, quant_img, num_colors)
# %%
from huggingface_hub import login

# login()
# %%
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
# stage_1.enable_model_cpu_offload()
# %%
# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
# stage_2.enable_model_cpu_offload()

# stage 3
# safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
# stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
# stage_3.enable_model_cpu_offload()
#
# prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'

# text embeds
# prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# generator = torch.manual_seed(0)

# stage 1
# image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
# pt_to_pil(image)[0].save("./if_stage_I.png")

# stage 2
# image = stage_2(
#     image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
# ).images
# pt_to_pil(image)[0].save("./if_stage_II.png")

# stage 3
# image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
# image[0].save("./if_stage_III.png")

# %%
quant_img_pil = Image.fromarray(quant_img)

# %%
unet_2 = stage_2.unet.to("cuda")
# %%
stage_1
# %%
scheduler = stage_1.scheduler

# Load the UNet model - the core denoiser
unet = stage_1.unet.to("cuda")

# Example: Create a noise tensor
image_shape = (1, 3, 64, 64)  # Batch size 1, 3 color channels, 64x64 image
noisy_image = torch.randn(image_shape, dtype=torch.float16, device="cuda")

# Predict denoised image using UNet
with torch.no_grad():
    noise_pred = unet(noisy_image, torch.tensor([1.0], device="cuda")).sample  # Predict noise

# Use scheduler to step back in denoising
denoised_image = scheduler.step(noise_pred, 1.0, noisy_image).prev_sample
# %%
class CosineNoiseScheduler:
    """
    Implements a cosine noise schedule for diffusion, as used in DeepFloyd IF.
    """
    def __init__(self, num_timesteps=1000, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.alphas_cumprod = self._cosine_schedule().to(device)

    def _cosine_schedule(self):
        """Generates a cosine noise schedule."""
        s = 0.008  # Small offset for numerical stability
        steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
        f = torch.cos(((steps / self.num_timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f / f[0]  # Normalize to start from 1
        return alphas_cumprod

    def step(self, noisy_image, t, noise_pred):
        """Performs one step of denoising using the cosine schedule."""
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
        beta_t = 1 - alpha_t / alpha_t_prev

        # Compute the denoised image
        denoised_image = (noisy_image - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        return denoised_image
# %%
cosine_scheduler = CosineNoiseScheduler(num_timesteps=1000, device=device)
# %%
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
class ViTImageEncoder(nn.Module):
    """
    Uses a pre-trained ViT model to extract embeddings from a quantized image.
    This replaces the CNN encoder with a stronger transformer-based encoder.
    """
    def __init__(self, model_name="google/vit-base-patch16-224", output_dim=1024):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.fc = nn.Linear(self.vit.config.hidden_size, output_dim)  # Resize to match UNet's expected size

    def forward(self, x):
        features = self.vit(x).last_hidden_state  # Extract token embeddings
        pooled_features = features.mean(dim=1)  # Global Average Pooling (B, D)
        return self.fc(pooled_features).unsqueeze(1)  # Shape: (batch, 1, output_dim)

# =============================
# 2️⃣ Load DeepFloyd IF UNet & Scheduler
# =============================


encoder = ViTImageEncoder(output_dim=unet.config.cross_attention_dim).to(device)
# %%
def run_dequantization(image, pipe):
    """
    Uses DeepFloyd IF Stage I for dequantization (restoring natural colors).
    
    Parameters:
    - image: PIL Image (quantized version)

    Returns:
    - Restored image as a PIL Image
    """
    # Load DeepFloyd IF Stage I model


    # Convert image to tensor (normalize to -1 to 1 for diffusion models)
    image = image.resize((64, 64))  # DeepFloyd IF Stage I uses 64x64 inputs
    image_np = np.array(image).astype(np.float32) / 127.5 - 1  # Normalize
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).to("cuda")

    # Run inference
    with torch.no_grad():
        output = pipe(image=image_tensor).images[0]

    return output

# Load and quantize an image

quantized_image = quant_img_pil

# Run DeepFloyd IF Stage I for dequantization
restored_image = run_dequantization(quantized_image, stage_2)
# %%
# Display results
fig, ax = plt.subplots(1, 3, figsize=(12, 5))
ax[0].imshow(original_image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(quantized_image)
ax[1].set_title("Quantized Image (16 colors)")
ax[1].axis("off")

ax[2].imshow(restored_image)
ax[2].set_title("Dequantized Image (DeepFloyd IF Stage I)")
ax[2].axis("off")

plt.show()
# %%
# %%
# %%
# %%
# %%
