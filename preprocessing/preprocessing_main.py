# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/preprocessing/"
os.chdir(new_directory)
# %%
from patient_loader import get_patients

# %%
patients = get_patients()
# %%
