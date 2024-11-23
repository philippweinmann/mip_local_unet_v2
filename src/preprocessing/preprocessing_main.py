# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)

# %%
from pathlib import Path
from src.patient_loader import get_patients
import nibabel as nib
import numpy as np
from src.preprocessing.preprocessing_utils import pad_image, resample_image, clip_scans, min_max_normalize, divide_3d_image_into_patches
from src.preprocessing import preprocessing_config

# %%
patients = get_patients()
# %%
# let_s start with a single patient
patient_x = patients[0]

def save_patches_for_patient(patient):
    image = nib.load(patient.image_fp)
    mask = nib.load(patient.mask_fp)

    # fix voxel spacing
    original_spacing = image.header.get_zooms()[:3]
    image = resample_image(image.get_fdata(), original_spacing=original_spacing, target_spacing=preprocessing_config.TARGET_VOXEL_SPACING)
    mask = resample_image(mask.get_fdata(), original_spacing=original_spacing, target_spacing=preprocessing_config.TARGET_VOXEL_SPACING)

    # clip image
    image = clip_scans(image, preprocessing_config.CLIPPING_MIN, preprocessing_config.CLIPPING_MAX)

    # min/max normalize
    image = min_max_normalize(image, preprocessing_config.CLIPPING_MIN, preprocessing_config.CLIPPING_MAX)

    # pad the image and mask as preparation for patching
    image, _ = pad_image(image, patch_size = preprocessing_config.PATCH_SIZE)
    mask, _ = pad_image(mask, patch_size = preprocessing_config.PATCH_SIZE)

    # divide the 3d image and mask into patches:
    block_shape = (preprocessing_config.PATCH_SIZE, preprocessing_config.PATCH_SIZE, preprocessing_config.PATCH_SIZE)
    image_patches = divide_3d_image_into_patches(padded_image, block_shape)
    mask_patches = divide_3d_image_into_patches(padded_mask, block_shape)

    # let's save them to disk
    image_patch_shape = image_patches.shape
    for x_dim in range(image_patch_shape[0]):
        for y_dim in range(image_patch_shape[1]):
            for z_dim in range(image_patch_shape[2]):
                current_image_patch = image_patches[x_dim, y_dim, z_dim]
                current_mask_patch = mask_patches[x_dim, y_dim, z_dim]

                np.savez(preprocessing_config.Output_dir / f"{patient.idx}_image_and_mask_patch_{x_dim}_{y_dim}_{z_dim}.npz", image = current_image_patch, mask = current_mask_patch)

save_patches_for_patient(patient=patient_x)

# %%
