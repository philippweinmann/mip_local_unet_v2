# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)

# %%
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from src.preprocessing import preprocessing_config


# %%
from src.patient_loader import get_patients

# %%
patients = get_patients()
# %%
patient_x = patients[0]
# %%
image = nib.load(patient_x.image_fp)
mask = nib.load(patient_x.label_fp)
# %%
print(image)
# %%
image.header.get_zooms()[:3]
# %%
x_dim_spacings = []
y_dim_spacings = []
z_dim_spacings = []

def get_all_voxel_spacings(image_fp):
    image = nib.load(image_fp)
    (x_dim_spacing, y_dim_spacing, z_dim_spacing) = image.header.get_zooms()[:3]
    x_dim_spacings.append(x_dim_spacing)
    y_dim_spacings.append(y_dim_spacing)
    z_dim_spacings.append(z_dim_spacing)

for patient in patients:
    get_all_voxel_spacings(patient.image_fp)
# %%
def plot_histogram(voxel_spacings, dim):
    plt.hist(voxel_spacings, bins=10)
    plt.title(f"voxel spacings in dimension: {dim}")
    plt.xlabel("bin means")
    plt.ylabel(f"voxel_spacings")
    plt.show()

voxel_spacings = [x_dim_spacings, y_dim_spacings, z_dim_spacings]
titles = ["x", "y", "z"]

for voxel_spacing, title in zip(voxel_spacings, titles):
    plot_histogram(voxel_spacing, dim=title)
# %%
import numpy as np
from scipy.ndimage import zoom

def get_voxel_spacing(patient):
    image_fp = patient.image_fp
    mask_fp = patient.label_fp
    
    image = nib.load(image_fp)
    mask = nib.load(mask_fp)
    (img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing) = image.header.get_zooms()[:3]
    (mask_x_dim_spacing, mask_y_dim_spacing, mask_z_dim_spacing) = mask.header.get_zooms()[:3]

    assert img_x_dim_spacing == mask_x_dim_spacing
    assert img_y_dim_spacing == mask_y_dim_spacing
    assert img_z_dim_spacing == mask_z_dim_spacing

    return [img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing]

voxel_spacings = get_voxel_spacing(patients[0])
target_spacing = [0.325, 0.325, 0.5]

def resample_image(image, original_spacing, target_spacing = preprocessing_config.TARGET_VOXEL_SPACING):
    print(original_spacing)
    print(target_spacing)
    zoom_factors = [original_spacing[i] / target_spacing[i] for i in range(3)]
    resampled_image = zoom(image, zoom_factors, order=1)  # Linear interpolation
    return resampled_image

image = nib.load(patient_x.image_fp).get_fdata()
resampled_image = resample_image(image, voxel_spacings, target_spacing)
print("Original shape:", image.shape)
print("Resampled shape:", resampled_image.shape)
# %%
