# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/src/preprocessing/"
os.chdir(new_directory)
# %%
from patient_loader import get_patients

# %%
patients = get_patients()
# %%
# let's start with a single patient
patient_x = patients[0]

image_fp = patient_x.image_fp
print(image_fp)
# %%
