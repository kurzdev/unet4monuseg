import torch
from torchvision.transforms import v2 as transforms


class OverlappingPatches(transforms.Transform):
    def __init__(self, patch_size: int, overlap: int):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:  # type: ignore
        patches = []
        rows, cols = img.shape[1], img.shape[2]
        step = self.patch_size - self.overlap

        for top in range(0, rows - self.patch_size + 1, step):
            for left in range(0, cols - self.patch_size + 1, step):
                patch = img[
                    :, top : top + self.patch_size, left : left + self.patch_size
                ]
                patches.append(patch)

        return patches
