# %%
import os
new_directory = "/home/tu-philw/group/gecko/pweinmann/mip_local_unet_v2/"
os.chdir(new_directory)
# %%
from sklearn.model_selection import train_test_split
import torch
from src.training_utils import get_preprocessed_patches, get_image_mask_from_patch_fp
from src.models.net_utils import get_best_device, prepare_image_for_network_input, calculate_dice_score
from src.models.unet import UNet3D, dice_bce_loss
import numpy as np
# %%
device = get_best_device()

if device == "mps":
    # mps is not supported for 3d
    device = "cpu"

torch.set_default_device(device)
print(f"Using {device} device. Every tensor created will be by default on {device}")
# %%
patch_fps = get_preprocessed_patches()
train, test = train_test_split(patch_fps, test_size=0.2, random_state=42)
# %%

# %%
print("----------------TRAINING-------------")

training_losses = []
def train_loop(model, loss_fn, optimizer, training_patches):
    model.train()

    avrg_loss = 0

    for idx, training_patch in enumerate(training_patches):
        image, mask = get_image_mask_from_patch_fp(training_patch)

        # set the learning rate here
        positive_voxels = np.sum(mask)
        learning_rate = max(0.005, min((positive_voxels / 100000), 0.2))

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        image = prepare_image_for_network_input(image)
        mask = prepare_image_for_network_input(mask)

        
        optimizer.zero_grad()

        prediction = model(image)

        loss = loss_fn(prediction, mask)
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        avrg_loss += loss_value

        dice = "no positive voxels"
        if positive_voxels > 0:
            dice = calculate_dice_score(prediction, mask)
            print(f"dice score: {dice}")

        print(f"loss: {loss_value:6f}, learning rate: {learning_rate}, dice: {dice}")


test_losses = []

def test_loop(model, loss_fn, test_patches):
    model.eval()

    avrg_loss = 0
    dice_avg = 0
    with torch.no_grad():
        for idx, test_patch in enumerate(test_patches):
            image, mask = get_image_mask_from_patch_fp(test_patch)

            image = prepare_image_for_network_input(image)
            mask = prepare_image_for_network_input(mask)

            prediction = model(image)

            loss = loss_fn(prediction, mask)
            
            test_losses.append(loss.item())
            avrg_loss += loss.item()
            dice_avg = calculate_dice_score(prediction, mask)

            if idx % 10 == 0:
                print(f"loss of last 10 patches: {avrg_loss / 10:8f}, dice score: {dice_avg/ 10:8f}")
                avrg_loss = 0
                dice_avg = 0
# %%
model = UNet3D(in_channels=1, num_classes=1)
model.to(device);

loss_fn = dice_bce_loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 2

try:
    for epoch in range(epochs):
        train_loop(model, loss_fn, optimizer, train)
        test_loop(model, loss_fn, test)
except KeyboardInterrupt:
    print("Training interrupted")
    model.eval()
# %%

# %%
# %%
