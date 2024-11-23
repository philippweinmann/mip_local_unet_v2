# %%
import os

from src.preprocessing.preprocessing_utils import get_voxel_spacing
from src.preprocessing.preprocessing_utils import resample_image
from visualizations.visualizations_utils import visualize_ccta_scan_slice
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)

# %%
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


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

voxel_spacings = get_voxel_spacing(patients[0])

image = nib.load(patient_x.image_fp).get_fdata()
resampled_image = resample_image(image, voxel_spacings)
print("Original shape:", image.shape)
print("Resampled shape:", resampled_image.shape)
visualize_ccta_scan_slice(image)

# %%
visualize_ccta_scan_slice(resampled_image)
# %%
