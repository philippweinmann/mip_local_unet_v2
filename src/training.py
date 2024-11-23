# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)
# %%
from sklearn.model_selection import train_test_split
import torch
from src.training_utils import get_preprocessed_patches

# %%
patch_fps = get_preprocessed_patches()
train, test = train_test_split(patch_fps, test_size=0.2, random_state=42)