import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
class ViTImageEncoder(nn.Module):
    """
    Uses a pre-trained ViT model to extract embeddings from a quantized image.
    This replaces the CNN encoder with a stronger transformer-based encoder.
    """
    def __init__(self,channels_in,  model_name="google/vit-base-patch16-224", output_dim=1024):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.fc = nn.Linear(self.vit.config.hidden_size, output_dim)  # Resize to match UNet's expected size
        self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(channels_in, 768, kernel_size=(16, 16), stride=(16, 16))
        self.vit.config.num_channels = channels_in
        self.vit.embeddings.patch_embeddings.num_channels = channels_in
    
    def forward(self, x):
        features = self.vit(x).last_hidden_state  # Extract token embeddings
        pooled_features = features.mean(dim=1)  # Global Average Pooling (B, D)
        return self.fc(pooled_features).unsqueeze(1)  # Shape: (batch, 1, output_dim)
