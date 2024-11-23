# %%
from src.preprocessing import preprocessing_config
import nibabel as nib
from scipy.ndimage import zoom

def get_voxel_spacing(image_fp):
    image = nib.load(image_fp)
    (img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing) = image.header.get_zooms()[:3]
    
    return [img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing]


def resample_image(image, original_spacing, target_spacing = preprocessing_config.TARGET_VOXEL_SPACING):
    print(f"original spacing: {original_spacing}")
    zoom_factors = [original_spacing[i] / target_spacing[i] for i in range(3)]
    resampled_image = zoom(image, zoom_factors, order=1)  # Linear interpolation
    print(f"file has now isotropopic voxel spacing: {target_spacing}")
    return resampled_image

def clip_scans(image, min_value, max_value):
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image