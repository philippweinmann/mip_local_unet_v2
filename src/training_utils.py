# %%
from src.configs import config

# %%
patches_folder = config.DATA_FOLDER
def get_preprocessed_patches():
    patch_fps = list(patches_folder.iterdir())

    print("amt of detected patch files: ", len(patch_fps))

    return patch_fps