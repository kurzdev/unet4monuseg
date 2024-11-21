import torch
from torchvision.transforms import v2 as transforms


class StitchPatches(transforms.Transform):
    def __init__(self, original_shape: tuple[int, int], patch_size: int, overlap: int):
        super().__init__()
        self.original_shape = original_shape
        self.patch_size = patch_size
        self.overlap = overlap

    def forward(self, patches: list[torch.Tensor]) -> torch.Tensor:  # type: ignore
        rows, cols = self.original_shape[0], self.original_shape[1]
        step = self.patch_size - self.overlap
        stitched_image = torch.zeros(rows, cols, dtype=patches[0].dtype)

        patch_idx = 0
        for top in range(0, rows - self.patch_size + 1, step):
            for left in range(0, cols - self.patch_size + 1, step):
                patch = patches[patch_idx]
                stitched_image[
                    top : top + self.patch_size, left : left + self.patch_size
                ] = patch
                patch_idx += 1

        return stitched_image
