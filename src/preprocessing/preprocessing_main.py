# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)
# %%
from pathlib import Path
# %%
from src.patient_loader import get_patients
import nibabel as nib
import numpy as np
from src.preprocessing.preprocessing_utils import resample_image, clip_scans, min_max_normalize, divide_3d_image_into_patches
from src.preprocessing import preprocessing_config

# %%
patients = get_patients()
# %%
# let_s start with a single patient
patient_x = patients[0]
# %%
image = nib.load(patient_x.image_fp)
mask = nib.load(patient_x.label_fp)

# %%
# resample the image and the mask to isotropic voxel spacing
original_spacing = image.header.get_zooms()[:3]

resampled_image = resample_image(image.get_fdata(), target_spacing=preprocessing_config.TARGET_VOXEL_SPACING, original_spacing=original_spacing)
resampled_mask = resample_image(mask.get_fdata(), target_spacing=preprocessing_config.TARGET_VOXEL_SPACING, original_spacing=original_spacing)
# %%
# clip the values of the image
clipped_resampled_image = clip_scans(resampled_image, preprocessing_config.CLIPPING_MIN, preprocessing_config.CLIPPING_MAX)
# %%
# min_max normalize the image
min_max_normalized_image = min_max_normalize(clipped_resampled_image, preprocessing_config.CLIPPING_MIN, preprocessing_config.CLIPPING_MAX)

# %%
def pad_image(image, patch_size):
    shape = image.shape
    print(shape)

    padded_image = image
    for idx, shape_dim in enumerate(shape):
        print(f"idx: {idx}")
        rest = shape_dim % patch_size

        if rest == 0:
            print(f"no padding required for dim {idx}. Original image shape: {image.shape}")
            continue

        # we need to add that many slices
        pad_length = patch_size - rest

        # Pad the array with zeros along the first dimension, the zeroes are added at the end
        pad_width = [(0, pad_length) if i == idx else (0, 0) for i in range(len(shape))]
        padded_image = np.pad(padded_image, pad_width, mode='constant')

        assert np.all(padded_image[-pad_length:0] == 0)

    return padded_image
# %%
# Pad the image to prepare for patch extraction
patch_size = 128
padded_image = pad_image(min_max_normalized_image, patch_size)
padded_mask = pad_image(resampled_mask, patch_size)

# %%
# divide the 3d image into patches
block_shape = (patch_size, patch_size, patch_size) # isomorphic patches
print(f"block_shape: {block_shape}")
print(f"padded_image shape: {padded_image.shape}")
image_patches = divide_3d_image_into_patches(padded_image, block_shape)
mask_patches = divide_3d_image_into_patches(padded_mask, block_shape)

# %%
# let's save the patches to disk
output_dir = preprocessing_config.Output_dir
output_dir.mkdir(parents=True, exist_ok=True)

img_patch_prefix = f"{patient_x.idx}_image_patch_"
label_patch_prefix = f"{patient_x.idx}_label_patch_"

image_patch_shape = image_patches.shape
for x_dim in range(image_patch_shape[0]):
    for y_dim in range(image_patch_shape[1]):
        for z_dim in range(image_patch_shape[2]):
            current_image_patch = image_patches[x_dim, y_dim, z_dim]
            current_mask_patch = mask_patches[x_dim, y_dim, z_dim]

            np.savez(output_dir / f"{patient_x.idx}_image_and_mask_patch_{x_dim}_{y_dim}_{z_dim}.npz", image = current_image_patch, mask = current_mask_patch)

