# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)
# %%
from sklearn.model_selection import train_test_split
import torch
from src.training_utils import get_preprocessed_patches
from src.models.net_utils import get_best_device
from src.models.unet import UNet3D
# %%
device = get_best_device()

if device == "mps":
    # mps is not supported for 3d
    device = "cpu"

torch.set_default_device(device)
print(f"Using {device} device. Every tensor created will be by default on {device}")
# %%
patch_fps = get_preprocessed_patches()
train, test = train_test_split(patch_fps, test_size=0.2, random_state=42)
# %%
model = UNet3D(in_channels=1, num_classes=1)
model.to(device);
# %%
print("----------------TRAINING-------------")

def train_loop(model, train_patch_fps, loss_fn, optimizer):
    pass
        