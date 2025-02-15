from fastai.vision.all import *

def get_imagenette_dataloader(data_dir="/mnt/wd/datasets/Imagenette", batch_size=32, image_size=320, num_workers=4):

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='val'),
        get_y=parent_label,
        item_tfms=Resize(image_size),
        batch_tfms=aug_transforms(size=image_size, min_scale=0.75)  # Includes common augmentations
    )

    # Create the DataLoaders
    dls = dblock.dataloaders(data_dir, bs=batch_size, num_workers=num_workers)

    return dls

