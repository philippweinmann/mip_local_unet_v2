# %%
from src.preprocessing import preprocessing_config
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.util import view_as_blocks

def get_voxel_spacing(image_fp):
    image = nib.load(image_fp)
    (img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing) = image.header.get_zooms()[:3]
    
    return [img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing]


def resample_image(image, original_spacing, target_spacing = preprocessing_config.TARGET_VOXEL_SPACING):
    zoom_factors = [original_spacing[i] / target_spacing[i] for i in range(3)]
    resampled_image = zoom(image, zoom_factors, order=1)  # Linear interpolation
    return resampled_image

def clip_scans(image, min_value, max_value):
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image

def min_max_normalize(image, min_value, max_value):
    image = (image - min_value) / (max_value - min_value)

    return image

def pad_image(image, patch_size):
    shape = image.shape
    print(shape)

    padded_image = image
    pad_lengths = []
    for idx, shape_dim in enumerate(shape):
        rest = shape_dim % patch_size

        if rest == 0:
            print(f"no padding required for dim {idx}. Original image shape: {image.shape}")
            continue

        # we need to add that many slices
        pad_length = patch_size - rest
        pad_lengths.append(pad_length)

        # Pad the array with zeros along the first dimension, the zeroes are added at the end
        pad_width = [(0, pad_length) if i == idx else (0, 0) for i in range(len(shape))]
        padded_image = np.pad(padded_image, pad_width, mode='constant')

        assert np.all(padded_image[-pad_length:0] == 0)

    return padded_image, pad_lengths

def divide_3d_image_into_patches(image_3d, block_shape):
    patches = view_as_blocks(image_3d, block_shape).squeeze()

    return patches