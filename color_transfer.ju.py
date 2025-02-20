# %%
import numpy as np
import torch
import torch.nn as nn
import math
import fastai
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from dataloader import get_imagenette_dataloader
from quantize import quantize_img, plot_imgs
from ddpm import DDPMCB
from preprocessing import clip_preprocess, conditioning_transform
from functools import partial
from fastai.vision.all import (ImageDataLoaders, Resize, TensorImage, Learner, 
                               Callback, Normalize)
from encoder import ViTImageEncoder
import fastcore.all as fc

device = "cuda"

# %%
def method_helper(o): return list(filter(lambda x: x[0] != "_", dir(o)))

# %%
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", 
     torch_dtype=torch.float16, class_labels=None 
)

# %%
scheduler = stage_2.scheduler
unet = stage_2.unet.to(device)

# %%
dls = ImageDataLoaders.from_folder( "/mnt/wd/datasets/imagenette2", valid_pct=0.1, bs=1,)
one_batch = dls.one_batch()[0]
one_batch.shape

# %%
encoder = ViTImageEncoder(7, output_dim=unet.config.encoder_hid_dim).to(device)
encoder_preprocess = encoder.feature_extractor
c_preprocess = partial(clip_preprocess, stage_2=stage_2)
cond_transform = partial(conditioning_transform, encode_preprocess=None)
# %%
def preprocessing(x):
    if not isinstance(x, fastai.vision.core.TensorCategory):
        x = Resize(224)(x)
        x = TensorImage(x).permute(2,1,0)
    x = cond_transform(x)
    x = x.to("cpu")
    x = c_preprocess(x)
    return x

# %%
from fastcore.foundation import store_attr
class SelectiveNormalize(Normalize):
    """Normalize only selected channels, keeping others unchanged."""
    def __init__(self, norm_channels=slice(3, None), mean=None, std=None, axes=(0,2,3)):
        store_attr()
        
    def setups(self, dl):
        """Compute mean & std only for selected channels if not provided."""
        if self.mean is None or self.std is None:
            x, *_ = dl.one_batch()
            full_mean = x.mean(self.axes, keepdim=True)
            full_std = x.std(self.axes, keepdim=True) + 1e-7

            self.mean = torch.zeros_like(full_mean)
            self.std = torch.ones_like(full_std)
            self.mean[:, self.norm_channels, :, :] = full_mean[:, self.norm_channels, :, :]
            self.std[:, self.norm_channels, :, :] = full_std[:, self.norm_channels, :, :]
# %%
dls = ImageDataLoaders.from_folder(
    "/mnt/wd/datasets/imagenette2",
    valid_pct=0.1,
    item_tfms=[preprocessing],
    # batch_tfms=[SelectiveNormalize()],
    bs=4,
    num_workers=16
)
# %%
dls.one_batch()[0].shape
dls.one_batch()[0][0,2,...].std()

# %%
class CTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = unet
        self.unet.class_embedding = None
        self.vit = ViTImageEncoder(7, output_dim=self.unet.config.encoder_hid_dim).to(device)

        for param in self.unet.parameters():
            param.requires_grad = False
        

    def forward(self, noisy_images, images, t):
        encoded = self.vit(images).expand(-1, 77, -1).half()

        return self.unet(noisy_images.half(), t.half(), encoded.half())[0]

# %%
model = CTModel()

# %%
one_batch = dls.one_batch()
one_batch[0].shape
images = one_batch[0]
# images = torch.cat([images, images], dim=1)

# %%
# Without DDPM callback it won't work
# with torch.no_grad():
#     x = model(images, one_batch[0], torch.tensor([1.0]*4, dtype=torch.float16, device="cuda"))
# x

# %%
learn = Learner(dls, model, loss_func=torch.nn.MSELoss(), cbs=[DDPMCB(unet,scheduler)]).to_fp16()

# %%
# learn.lr_find()

# %%
lr = 10e-04
learn.fit_one_cycle(1, lr)

# %%
learn.save("ctransfer_epoch_1.pth")
# learn = learn.load("ctransfer_epoch_1.pth")

# %%


# %%
# learn2 = learn.add_cb(astats)

# %%


# %%
# astats.color_dim()

# %%
lr = 10e-05
# learn.fit_one_cycle(1, lr)
learn.load("ctransfer_epoch_2.pth")

# %%
lr = 10e-05
# learn.save("ctransfer_epoch_2.pth")
learn.fit_one_cycle(4, lr)
learn.save("ctransfer_epoch_3_6.pth")
# learn.save("ctransfer_epoch_4.pth")
# learn.fit_one_cycle(1, lr)
# learn.save("ctransfer_epoch_5.pth")

# %%
learn.fit_one_cycle(3, lr)
learn.save("ctransfer_epoch_6_8.pth")

# %%
lr = 10e-06
learn.fit_one_cycle(1, lr)
learn.save("ctransfer_epoch_9.pth")

