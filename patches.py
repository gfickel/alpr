import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_patches(patches, title):
    return
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i < patches.shape[0]:
            ax.imshow(patches[i].permute(1, 2, 0).cpu().numpy())
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Create a sample image
image = torch.randn(1, 3, 32, 128)  # [batch_size, channels, height, width]
print("Original image shape:", image.shape)  # Expected: [1, 3, 32, 128]

patch_height, patch_width = 32, 8

# Test 1: nn.Unfold without overlap
unfold_no_overlap = nn.Unfold(kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width))
patches_unfold_no_overlap = unfold_no_overlap(image)
# Shape after unfold: [batch_size, channels * kernel_height * kernel_width, num_patches]
print("Shape after nn.Unfold (no overlap):", patches_unfold_no_overlap.shape)  # Expected: [1, 768, 16]

patches_unfold_no_overlap = patches_unfold_no_overlap.transpose(1, 2)
# Shape after transpose: [batch_size, num_patches, channels * kernel_height * kernel_width]
print("Shape after transpose:", patches_unfold_no_overlap.shape)  # Expected: [1, 16, 768]

patches_unfold_no_overlap = patches_unfold_no_overlap.reshape(-1, 3, patch_height, patch_width)
# Final shape: [num_patches, channels, patch_height, patch_width]
print("Final shape (nn.Unfold without overlap):", patches_unfold_no_overlap.shape)  # Expected: [16, 3, 32, 8]
plot_patches(patches_unfold_no_overlap, "nn.Unfold without overlap")

# Test 2: nn.Unfold with overlap
unfold_overlap = nn.Unfold(kernel_size=(patch_height, patch_width), stride=(patch_height // 2, patch_width // 2))
patches_unfold_overlap = unfold_overlap(image)
# Shape after unfold: [batch_size, channels * kernel_height * kernel_width, num_patches]
print("Shape after nn.Unfold (with overlap):", patches_unfold_overlap.shape)  # Expected: [1, 768, 45]

patches_unfold_overlap = patches_unfold_overlap.transpose(1, 2)
# Shape after transpose: [batch_size, num_patches, channels * kernel_height * kernel_width]
print("Shape after transpose:", patches_unfold_overlap.shape)  # Expected: [1, 45, 768]

patches_unfold_overlap = patches_unfold_overlap.reshape(-1, 3, patch_height, patch_width)
# Final shape: [num_patches, channels, patch_height, patch_width]
print("Final shape (nn.Unfold with overlap):", patches_unfold_overlap.shape)  # Expected: [45, 3, 32, 8]
plot_patches(patches_unfold_overlap, "nn.Unfold with overlap")

# Comparison
print("\nComparison of final shapes:")
print("nn.Unfold without overlap:", patches_unfold_no_overlap.shape)
print("nn.Unfold with overlap:", patches_unfold_overlap.shape)