import torch

# Implementation based on https://github.com/milesial/Pytorch-UNet/tree/master


def dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - dice_coeff(input, target, reduce_batch_first=True)
