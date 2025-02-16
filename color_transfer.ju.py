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
from fastai.vision.all import ImageDataLoaders, Resize, TensorImage, Learner, Callback
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
cond_transform(img).shape
# %%
def preprocessing(x):
    if isinstance(x, fastai.vision.core.PILImage):
        x = TensorImage(x).permute(2,1,0).numpy()
    x = c_preprocess(x)
    return cond_transform(x)

preprocessing(one_batch[0].cpu().numpy()).shape
# %%
dls = ImageDataLoaders.from_folder(
    "/mnt/wd/datasets/imagenette2",
    valid_pct=0.1,
    item_tfms=[Resize(224), preprocessing],
    bs=4,
    num_workers=16
)

# %%
dls.one_batch()[0].shape


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
# learn = Learner(dls, model.half(), loss_func=torch.nn.MSELoss(), cbs=[DDPMCB(unet,scheduler)])
# from fastai.learner import AvgSmoothLoss

# class FP16AvgSmoothLoss(AvgSmoothLoss):
#     def accumulate(self, learn):
#         self.count += 1
#         loss_fp16 = to_detach(learn.loss.mean()).half()  # Ensure FP16
#         self.val = torch.lerp(loss_fp16, self.val.half(), self.beta)  # Convert self.val to FP16

# learn.recorder.metrics = []

# learn.lr_find()

# %%
lr = 10e-05
# learn.fit_one_cycle(1, lr)
# learn.save("ctransfer_epoch_1.pth")
learn = learn.load("ctransfer_epoch_1.pth")
# %%
from collections.abc import Mapping
class Hook():
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


class Hooks(list):
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()
    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
    def remove(self):
        for h in self: h.remove()


class HooksCallback(Callback):
    def __init__(self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None):
        fc.store_attr()
        super().__init__()
    
    def before_fit(self):
        learn = self.learn
        if self.mods: mods=self.mods
        else: mods = fc.filter_ex(learn.model.modules(), self.mod_filter)
        self.hooks = Hooks(mods, partial(self._hookfunc, learn))

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training): self.hookfunc(*args, **kwargs)

    def after_fit(self): self.learn.hooks.remove()
    def __iter__(self): return iter(self.hooks)
    def __len__(self): return len(self.hooks)

def to_cpu(x):
    if isinstance(x, Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    acts = to_cpu(outp)
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40,0,10))

class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop): 
        super().__init__(hookfunc=append_stats, mod_filter=mod_filter)

    def color_dim(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin='lower')

    def dead_chart(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0,1)

    def plot_stats(self, figsize=(10,4)):
        fig,axs = plt.subplots(1,2, figsize=figsize)
        for h in self:
            for i in 0,1: axs[i].plot(h.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title('Stdevs')
        plt.legend(fc.L.range(self))
        
astats = ActivationStats(fc.risinstance("Linear"))
# %%
learn2 = learn.add_cb(astats)
learn2.save("ctransfer_epoch_2.pth")
# %%
import matplotlib.pyplot as plt
def subplots(
    nrows:int=1, # Number of rows in returned axes grid
    ncols:int=1, # Number of columns in returned axes grid
    figsize:tuple=None, # Width, height in inches of the returned figure
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure
    suptitle:str=None, # Title to be set to returned figure
    **kwargs
): # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None: figsize=(ncols*imsize, nrows*imsize)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows*ncols==1: ax = np.array([ax])
    return fig,ax


def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ('cpu','permute','detach')):
        im = im.detach().cpu()
        if len(im.shape)==3 and im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=np.array(im)
    if im.shape[-1]==1: im=im[...,0]
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    if noframe: ax.axis('off')
    return ax

def get_grid(
    n:int, # Number of axes
    nrows:int=None, # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols:int=None, # Number of columns, defaulting to `ceil(n/rows)`
    title:str=None, # If passed, title set to the figure
    weight:str='bold', # Title font weight
    size:int=14, # Title font size
    **kwargs,
): # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows: ncols = ncols or int(np.floor(n/nrows))
    elif ncols: nrows = nrows or int(np.ceil(n/ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n/nrows))
    fig,axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows*ncols): axs.flat[i].set_axis_off()
    if title is not None: fig.suptitle(title, weight=weight, size=size)
    return fig,axs


def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()

# %%
astats.color_dim()
# %%
# %%
