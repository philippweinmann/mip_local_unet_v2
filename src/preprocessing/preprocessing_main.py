# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)
# %%
from pathlib import Path
# %%
from src.patient_loader import get_patients
import nibabel as nib
from src.preprocessing.preprocessing_utils import resample_image, clip_scans, min_max_normalize, pad_image, divide_3d_image_into_patches
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
# Pad the image to prepare for patch extraction
patch_size = 128
padded_image = pad_image(min_max_normalized_image, patch_size)
padded_mask = pad_image(resampled_mask, patch_size)

# %%
# divide the 3d image into patches
block_shape = (patch_size, patch_size, patch_size) # isomorphic patches
image_patches = divide_3d_image_into_patches(padded_image, block_shape)
mask_patches = divide_3d_image_into_patches(padded_mask, block_shape)

# %%
# let's save the patches to disk
output_dir = preprocessing_config.Output_dir
img_patch_prefix = f"{patient_x.idx}_image_patch_"
label_patch_prefix = f"{patient_x.idx}_label_patch_"

for idx, (image_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
    img_patch_fp = output_dir / f"{img_patch_prefix}{idx}.nii.gz"
    label_patch_fp = output_dir / f"{label_patch_prefix}{idx}.nii.gz"

    img_patch = nib.Nifti1Image(image_patch, affine=image.affine)
    label_patch = nib.Nifti1Image(mask_patch, affine=mask.affine)

    nib.save(img_patch, img_patch_fp)
    nib.save(label_patch, label_patch_fp)

# %%
# Let's visualize the first patch
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(image_patches[0][64], cmap="gray")
plt.axis("off")
plt.show()

# %%
plt.imshow(mask_patches[0][64], cmap="gray")
plt.axis("off")
plt.show()
# %%
