import numpy as np


class TubeMaskingGenerator:
    """
    Generates tube-style masks for video patches.
    Space-time tubes: We mask the same spatial region across multiple frames (or all frames).
    """

    def __init__(self, input_size, patch_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.patch_size = patch_size
        self.num_patches_t = self.frames // patch_size[0] if len(patch_size) > 2 else self.frames
        self.num_patches_h = self.height // patch_size[-2]
        self.num_patches_w = self.width // patch_size[-1]
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w
        self.mask_ratio = mask_ratio

        # Determine number of masked patches
        self.num_masked = int(self.num_patches * mask_ratio)

    def __call__(self):
        # Simple random masking for now, TODO: True Tube Masking (Spatial block across time)
        # For true tube masking, we would sample spatial locations and extend them over time.

        mask = np.zeros(self.num_patches, dtype=int)
        mask_indices = np.random.choice(self.num_patches, self.num_masked, replace=False)
        mask[mask_indices] = 1
        return mask


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masked = int(self.num_patches * mask_ratio)

    def __call__(self):
        mask = np.hstack(
            [
                np.zeros(self.num_patches - self.num_masked, dtype=int),
                np.ones(self.num_masked, dtype=int),
            ]
        )
        np.random.shuffle(mask)
        return mask
