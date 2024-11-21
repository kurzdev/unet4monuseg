import argparse

import skimage.measure as measure
import torch

from lib.data import DataSubset, MoNuSegDataset
from lib.mask import dfs, skimage_patch, union_find
from lib.transform import quadrants
from lib.util.device import get_device
from lib.util.file import (
    DATA_DEFAULT_PATH,
    DATA_PATCHES_PATH,
)
from lib.util.model import load_unet
from lib.util.predict import predict_mask

PATCHES_PER_DIM = 7

weight_matrix = torch.full((PATCHES_PER_DIM + 1, PATCHES_PER_DIM + 1), 0.25)
weight_matrix[0, 0] = weight_matrix[0, -1] = 1
weight_matrix[-1, 0] = weight_matrix[-1, -1] = 1
weight_matrix[0, 1:-1] = weight_matrix[-1, 1:-1] = 0.5
weight_matrix[1:-1, 0] = weight_matrix[1:-1, -1] = 0.5


def count_cells_combined_patches(masks: list[torch.Tensor]) -> float:
    count_matrix = torch.zeros((PATCHES_PER_DIM + 1, PATCHES_PER_DIM + 1))

    for idx, mask in enumerate(masks):
        mask_quadrants = quadrants(mask)
        counts = torch.Tensor(
            [skimage_patch(quadrant) for quadrant in mask_quadrants]
        ).reshape((2, 2))

        row = idx // PATCHES_PER_DIM
        col = idx % PATCHES_PER_DIM
        count_matrix[row : row + 2, col : col + 2] += counts

    cells = count_matrix * weight_matrix
    return cells.sum().item()


def count_cells(mask: torch.Tensor) -> None:
    try:
        dfs_count = dfs(mask.clone())
        print(f"DFS count: {dfs_count}")
    except RecursionError:
        print("RecursionError occurred during DFS count")

    union_find_count = union_find(mask.clone())
    print(f"Union-Find count: {union_find_count}")

    _, skimage_count = measure.label(mask.clone().numpy(), return_num=True)  # type: ignore
    print(f"Skimage count: {skimage_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Count cells in (predicted) mask")
    parser.add_argument(
        "--timestamp",
        type=str,
        required=False,
        help="Timestamp of the model used to predict",
    )
    parser.add_argument(
        "--combine_patches",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Count cells for full size images by combining patches",
    )
    parser.add_argument(
        "--fullsize",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Use full size images and masks",
    )

    args = parser.parse_args()

    if args.combine_patches and args.fullsize:
        raise ValueError(
            "Cannot count using combined patches and full-sized images in the same run"
        )

    device = get_device()
    print(f"Using device: {device}")

    dataset = MoNuSegDataset(
        data_path=DATA_PATCHES_PATH if not args.fullsize else DATA_DEFAULT_PATH,
        data_subset=DataSubset.TEST,
    )

    image, mask, identifier = dataset[0]
    mask = mask.squeeze(0) > 0.5

    model = None

    if args.timestamp:
        model = load_unet(args.timestamp, device=device)
        mask = predict_mask(model, image, device)

    if args.combine_patches:
        patch_identifier = dataset.get_identifiers(without_patch_suffix=True)[0]
        count_matrix = torch.zeros((PATCHES_PER_DIM + 1, PATCHES_PER_DIM + 1))

        patches = dataset.get_patches_by_identifier(patch_identifier)
        masks = []
        for image, ground_truth, _ in patches:
            masks.append(
                predict_mask(model, image, device) > 0.5
                if model
                else ground_truth.squeeze(0) > 0.5
            )

        cells = int(count_cells_combined_patches(masks))
        print(f"Patch {patch_identifier}: {cells} cells")

    print(f"Image: {identifier}")
    count_cells(mask)
