import torch
from fastai.vision.all import TensorCategory
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, Resize
from torchvision.transforms import functional as TF
import cv2

def ignore_category(f):
    def wrapper(frame, **kwargs):
        if not isinstance(frame, TensorCategory):
            frame = f(frame, **kwargs)
        return frame
    return wrapper


def quantize_image(image: Image.Image, num_colors: int) -> Image.Image:
    """Quantizes a PIL image using the median cut algorithm."""
    return image.convert("RGB").quantize(colors=num_colors, method=Image.MEDIANCUT).convert("RGB")


def compute_luminance(image: Image.Image) -> Image.Image:
    """Computes the luminance (grayscale) channel of a PIL image."""
    return ImageOps.grayscale(image)


def compute_gradients(luminance: Image.Image) -> tuple:
    """Computes x- and y-direction gradients of the luminance channel."""
    lum_array = np.array(luminance).astype(np.float32)
    grad_x = cv2.Sobel(lum_array, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum_array, cv2.CV_32F, 0, 1, ksize=3)
    return grad_x, grad_y


def threshold_gradients(grad_x, grad_y, threshold=8) -> tuple:
    """Thresholds gradients to create binary images."""
    grad_x = (np.abs(grad_x) > threshold).astype(np.float32)
    grad_y = (np.abs(grad_y) > threshold).astype(np.float32)
    return grad_x, grad_y


@ignore_category
def conditioning_transform(frame: torch.Tensor,*, encode_preprocess) -> torch.Tensor:
    """
    Applies all transformatiqons to generate the 7-channel input.
    
    Args:
        frame (torch.Tensor): A tensor image of shape (C, H, W) in range [0, 1].
    
    Returns:
        torch.Tensor: A 7-channel tensor of shape (7, H, W) in range [0, 1].
    """
    # Convert tensor to PIL image for processing
    gt_frame, frame = torch.split(frame, 3, dim=0)
    image = TF.to_pil_image(frame)

    # 1-3: Quantized Image
    num_colors = 2 ** np.random.randint(2, 8)
    quantized_image = quantize_image(image, num_colors)
    quantized_tensor = TF.to_tensor(quantized_image)

    # 4: Quantization Level Channel (normalized to [0,1])
    quant_level_channel = torch.full((1, quantized_tensor.shape[1], quantized_tensor.shape[2]), num_colors / 256)

    # 5: Luminance Channel
    luminance = compute_luminance(image)
    luminance_tensor = TF.to_tensor(luminance)

    # 6: Gradient-Based Conditioning
    grad_x, grad_y = compute_gradients(luminance)
    grad_x_tensor = torch.tensor(grad_x, dtype=torch.float32).unsqueeze(0) / 255.0
    grad_y_tensor = torch.tensor(grad_y, dtype=torch.float32).unsqueeze(0) / 255.0

    # 7: Texture Indicator (1 if texture is present, 0 otherwise)
    texture_indicator = torch.ones_like(quant_level_channel)

    # Stack all channels into a 7-channel tensor
    stacked_tensor = torch.cat([quantized_tensor, quant_level_channel, grad_x_tensor, grad_y_tensor, texture_indicator], dim=0)

    return torch.cat([gt_frame.half(), stacked_tensor.half()])


@ignore_category
def clip_preprocess(frame,*, stage_2):
    # frame for the noise input and gt
    clip_processor = stage_2.feature_extractor
    gt_frame = torch.tensor(clip_processor(frame)["pixel_values"][0])

    # frame for the encoder
    frame = ToTensor()(frame)
    frame = Resize((224,224))(frame.permute(1,2,0))
    return torch.cat([gt_frame, frame], dim=0)  # Shape: (6, H, W)


    
@ignore_category
def to_f16(frame):
    torch.tensor(frame, dtype=torch.float16)
