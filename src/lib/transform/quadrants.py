import torch


def quadrants(mask: torch.Tensor, size: int = 128) -> torch.Tensor:
    return torch.stack(
        [
            mask[:size, :size],
            mask[:size, size:],
            mask[size:, :size],
            mask[size:, size:],
        ]
    )
