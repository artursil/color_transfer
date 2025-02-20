import torch
import torch.nn as nn
from fastai.vision.all import Callback


class DDPMCB(Callback):
    """Custom FastAI callback for training a UNet diffusion model with DeepFloyd's DDPM Scheduler."""

    def __init__(self, unet, scheduler, timesteps=1000):
        self.unet = unet
        self.scheduler = scheduler
        self.timesteps = timesteps  # Number of diffusion steps

    def before_batch(self):
        """Add noise to the input images before passing them to the model."""
        # Get the real images from the batch
        images = self.xb[0]  # Shape: (batch_size, 3, H, W)
        gt_imgs = images[:, :3, ...]
        enc_imgs = images[:, 3:, ...]
        batch_size = images.shape[0]

        # Generate noise
        t = torch.randint(0, self.timesteps, (batch_size,), device=images.device).long()
        noise = torch.randn_like(gt_imgs)

        # Apply noise
        noisy_images = self.scheduler.add_noise(gt_imgs, noise, t)

        noisy_images = torch.cat([noisy_images, noisy_images], dim=1)
        noise = torch.cat([noise, noise], dim=1)

        self.learn.xb = (noisy_images, enc_imgs, t)
        self.learn.yb = (noise,)

    def after_pred(self):
        """Compute loss: Train UNet to predict noise (MSE loss)."""
        gt_image_pred = self.pred.to(torch.float32)  # Model's prediction
        gt_image_true = self.yb[0]  # Ground truth (now 6-channel)

        self.learn.loss = nn.functional.mse_loss(gt_image_pred, gt_image_true)

