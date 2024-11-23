# %%
from src.configs import config
import torch
import numpy as np
# %%
patches_folder = config.DATA_FOLDER
def get_preprocessed_patches():
    patch_fps = list(patches_folder.iterdir())

    print("amt of detected patch files: ", len(patch_fps))

    return patch_fps

def get_image_mask_from_patch_fp(patch_fp):
    patch = np.load(patch_fp)
    image = patch["image"]
    mask = patch["mask"]

    return image, mask