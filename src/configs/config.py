from pathlib import Path

DATA_FOLDER = Path("/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/data/preprocessed_data/")

# training configurations
DICE_MULTIPLIER_HYPERPARAMETER = 1000
NUMBER_OF_POS_FOR_BCE_TO_BE_1 = 200000