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
# stage g
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", 
     torch_dtype=torch.float16, class_labels=None 
)
# stage_2.enable_model_cpu_offload()

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

# %%
img = c_preprocess(one_batch[0].cpu().numpy())
# %%
cond_transform = partial(conditioning_transform, encode_preprocess=None)
# cond_transform(img).shape
# %%
def preprocessing(x):
    if not isinstance(x, fastai.vision.core.TensorCategory):
        # x = TensorImage(x).permute(2,1,0).numpy()
        x = Resize(224)(x)
        x = TensorImage(x).permute(2,1,0)
    x = cond_transform(x)
    x = x.to("cpu")
    # x = c_preprocess(x)
    return x

# preprocessing(one_batch[0].cpu().numpy()).shape
# %%
# dls = ImageDataLoaders.from_folder(
#     "/mnt/wd/datasets/imagenette2",
#     valid_pct=0.1,
#     item_tfms=[preprocessing],
# #     batch_tfms=[Normalize()],
#     bs=3000,
#     num_workers=16
# )
# xb, _ = dls.one_batch()
# mean = xb.mean(dim=[0,2,3])  # Compute per-channel mean
# std = xb.std(dim=[0,2,3])    # Compute per-channel std
# print("Auto-calculated Mean:", mean)
# print("Auto-calculated Std:", std)

# %%
mean = torch.tensor([ 4.6301e-01,  4.5852e-01,  4.3105e-01,  1.8062e-03,  1.7940e-03,
                    1.6879e-03,  6.3991e-04, -1.0427e-05, -8.0591e-08,  3.9216e-03])
std = torch.tensor([0.2826, 0.2781, 0.3003, 0.0011, 0.0011, 0.0012, 0.0007, 0.0015, 0.0014, 0.0000])
# %%
dls = ImageDataLoaders.from_folder(
    "/mnt/wd/datasets/imagenette2",
    valid_pct=0.1,
    item_tfms=[preprocessing],
    # batch_tfms=[Normalize.from_stats(mean, std)],
    batch_tfms=[Normalize()],
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
from fastai.callback.hook import ActivationStats

# Create a list of layers to track. You can add or remove layers based on what you want to observe.
layers_to_track = [
    learn.model.vit.vit.embeddings.patch_embeddings.projection,
    learn.model.vit.vit.encoder.layer[0].attention.attention.query,
    learn.model.vit.vit.encoder.layer[0].attention.attention.key,
    learn.model.vit.vit.encoder.layer[0].attention.attention.value,
    learn.model.vit.vit.encoder.layer[0].intermediate.dense,
    learn.model.vit.vit.encoder.layer[0].output.dense,
    learn.model.vit.vit.encoder.layer[0].layernorm_before,
    learn.model.vit.vit.encoder.layer[0].layernorm_after,
    learn.model.vit.vit.encoder.layer[6].attention.attention.query,
    learn.model.vit.vit.encoder.layer[6].attention.attention.key,
    learn.model.vit.vit.encoder.layer[6].attention.attention.value,
    learn.model.vit.vit.encoder.layer[6].intermediate.dense,
    learn.model.vit.vit.encoder.layer[6].output.dense,
    learn.model.vit.vit.encoder.layer[6].layernorm_before,
    learn.model.vit.vit.encoder.layer[6].layernorm_after,
    learn.model.vit.vit.layernorm,
    learn.model.vit.vit.pooler.dense,
]

# Add the ActivationStats callback
astats = ActivationStats(modules=layers_to_track)
learn.add_cb(astats)

# %%
# learn.lr_find()
# %%
lr = 10e-05
learn.fit_one_cycle(1, lr)
learn.save("ctransfer_epoch_1.pth")
# learn = learn.load("ctransfer_epoch_1.pth")
# %%
        
# %%
learn2 = learn.add_cb(astats)
learn2.save("ctransfer_epoch_2.pth")
# %%

# %%
astats.color_dim()
# %%
# %%
