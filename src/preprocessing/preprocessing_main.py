# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)
# %%
from pathlib import Path
# %%
from src.patient_loader import get_patients
import nibabel as nib
from src.preprocessing.preprocessing_utils import resample_image, clip_scans
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
clipped_resampled_image = clip_scans(resampled_image, -600, 1000)
# %%
