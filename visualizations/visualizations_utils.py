# %%
import matplotlib.pyplot as plt


def visualize_ccta_scan_slice(image, slice_idx=60):
    plt.imshow(image[:, :, slice_idx], cmap="gray")
    plt.axis("off")
    plt.show()