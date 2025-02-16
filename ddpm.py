from fastai.vision.all import Callback
import torch
import torch.nn as nn

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
        batch_size = images.shape[0]

        # Generate noise
        t = torch.randint(0, self.timesteps, (batch_size,), device=images.device).long()
        noise = torch.randn_like(images)[:,:3,...]

        # Apply noise
        noisy_images = self.scheduler.add_noise(images[:,:3,...], noise, t)

        noisy_images = torch.cat([noisy_images, noisy_images], dim=1)  # Shape: (batch_size, 6, H, W)
        gt_image_true = torch.cat([images[:,:3,...], images[:,:3,...]], dim=1)  # Shape: (batch_size, 6, H, W)
 
        self.learn.xb = (noisy_images, images, t)
        self.learn.yb = (gt_image_true,)

    def after_pred(self):
        """Compute loss: Train UNet to predict noise (MSE loss)."""
        gt_image_pred = self.pred.to(torch.float32)  # Model's prediction
        gt_image_true = self.yb[0]  # Ground truth (now 6-channel)
        
        self.learn.loss = nn.functional.mse_loss(gt_image_pred, gt_image_true)

    # def after_batch(self):
    #     """Zero gradients manually after each batch (recommended for diffusion)."""
    #     self.learn.opt.zero_grad()
