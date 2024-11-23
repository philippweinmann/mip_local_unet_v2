# %%
from src.configs import config

# %%
patches_folder = config.DATA_FOLDER
def get_training_patches():
    patches_files = list(patches_folder.iterdir())

    print("amt of detected_files: ", len(patches_files))

    return patches_files