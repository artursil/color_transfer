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
from ddpm import DDPMCB
from preprocessing import clip_preprocess, conditioning_transform
from functools import partial
from fastai.vision.all import ImageDataLoaders
from encoder import ViTImageEncoder

device = "cuda"

# %%

def method_helper(o): return list(filter(lambda x: x[0] != "_", dir(o)))

# %%
encoder = ViTImageEncoder(7, output_dim=unet.config.encoder_hid_dim).to(device)
encoder_preprocess = encoder.feature_extractor
# %%
# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", 
     torch_dtype=torch.float16, class_labels=None 
)
# stage_2.enable_model_cpu_offload()

# %%
scheduler = stage_2.scheduler

# Load the UNet model - the core denoiser
unet = stage_2.unet.to(device)

# %%
clip_preprocess = partial(clip_preprocess, stage_2=stage_2)

# %%
dls = ImageDataLoaders.from_folder(
    "/mnt/wd/datasets/imagenette2",
    valid_pct=0.1,
    item_tfms=[clip_preprocess, conditioning_transform],
    bs=4,
    num_workers=16
)

# %%
dls.one_batch()[0].shape

# %%
# dls.one_batch()

# %%
# dls = ImageDataLoaders.from_folder("/mnt/wd/datasets/imagenette2", valid_pct=0.1, bs=4, item_tfms=Resize(224))

# %%
# def preprocess(frame)
#     return clip_processor(frame)["pixel_values"][0]

# %%
# clip_processor(dls.one_batch()[0], rescale=False)["pixel_values"][0].shape

# %%
one_batch = dls.one_batch()[0]

# %%


# %%



# %%
# %%

encoder(dls.one_batch()[0])

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
images = torch.cat([images, images], dim=1)

# %%
with torch.no_grad():
    x = model(images, one_batch[0], torch.tensor([1.0]*4, dtype=torch.float16, device="cuda"))
# x

# %%
x[0].size()

# %%
learn = Learner(dls, model, loss_func=torch.nn.MSELoss(), cbs=[DDPMCB(unet,scheduler)]).to_fp16()
# learn = Learner(dls, model.half(), loss_func=torch.nn.MSELoss(), cbs=[DDPMCB(unet,scheduler)])
# from fastai.learner import AvgSmoothLoss

# class FP16AvgSmoothLoss(AvgSmoothLoss):
#     def accumulate(self, learn):
#         self.count += 1
#         loss_fp16 = to_detach(learn.loss.mean()).half()  # Ensure FP16
#         self.val = torch.lerp(loss_fp16, self.val.half(), self.beta)  # Convert self.val to FP16

# learn.recorder.metrics = []

learn.lr_find()

# %%
lr = 10e-05
learn.fit_one_cycle(1, lr)

# %%
# If lr_max is not provided, use the suggested learning rate from the finder
    lr_max = lr_max or lr_max_suggested
    print(f"Using learning rate: {lr_max:.2e}")

    # 🚀 Step 2: Train the model with OneCycle policy
    learn.fit_one_cycle(epochs, lr_max)

    return learn  # Return trained Learner

# %%
x = x.expand(-1, 77, -1)
one_batch = torch.cat([one_batch, one_batch], dim=1)

