from pathlib import Path

import numpy as np
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

from lib.data import DataSubset, MoNuSegDataset
from lib.parse import parse_mask_xml
from lib.transform import OverlappingPatches
from lib.util.file import (
    DATA_DEFAULT_PATH,
    DATA_PATCHES_PATH,
    image_filename,
    parsed_mask_filename,
    patch_suffix,
)


def parse_masks(dataset: MoNuSegDataset) -> None:
    identifiers = dataset.get_identifiers()

    print(f"Got {len(identifiers)} identifiers")

    for identifier in tqdm(identifiers, desc="Parsing masks"):
        image = dataset.load_image(identifier=identifier)
        mask_xml = dataset.load_mask_xml(identifier=identifier)
        binary_mask, color_mask = parse_mask_xml(
            mask_xml,
            nrow=image.shape[1],
            ncol=image.shape[2],
        )

        print(f"Parsed mask for identifier {identifier}!")
        print(f"Binary mask shape: {binary_mask.shape}")
        print(f"Color mask shape: {color_mask.shape}")

        np.save(
            dataset.data_path
            / dataset.data_subset.value
            / parsed_mask_filename(identifier, color=False),
            binary_mask.numpy(),
        )
        np.save(
            dataset.data_path
            / dataset.data_subset.value
            / parsed_mask_filename(identifier, color=True),
            color_mask.numpy(),
        )


def create_patches(
    dataset: MoNuSegDataset,
    patched_data_path: Path,
    resize_size: tuple[int, int] = (1024, 1024),
    patch_size: int = 256,
    overlap: int = 128,
) -> None:
    identifiers = dataset.get_identifiers()
    print(f"Got {len(identifiers)} identifiers")

    for identifier in tqdm(identifiers, desc="Creating patches"):
        image = dataset.load_image(identifier)
        binary_mask = dataset.load_mask(identifier, color=False)
        color_mask = dataset.load_mask(identifier, color=True)

        transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                OverlappingPatches(patch_size, overlap),
            ]
        )

        image_patches = transform(image)
        binary_mask_patches = transform(binary_mask)
        color_mask_patches = transform(color_mask)

        for i, (image_patch, binary_mask_patch, color_mask_patch) in enumerate(
            zip(image_patches, binary_mask_patches, color_mask_patches)
        ):
            image_patch_filename = (
                patched_data_path
                / dataset.data_subset.value
                / image_filename(f"{identifier}{patch_suffix(i)}")
            )
            binary_mask_patch_filename = (
                patched_data_path
                / dataset.data_subset.value
                / parsed_mask_filename(
                    identifier=f"{identifier}{patch_suffix(i)}", color=False
                )
            )
            color_mask_patch_filename = (
                patched_data_path
                / dataset.data_subset.value
                / parsed_mask_filename(
                    identifier=f"{identifier}{patch_suffix(i)}", color=True
                )
            )

            F.to_pil_image(image_patch).save(image_patch_filename)
            np.save(binary_mask_patch_filename, binary_mask_patch.numpy())
            np.save(color_mask_patch_filename, color_mask_patch.numpy())


if __name__ == "__main__":
    default_dataset_train = MoNuSegDataset(
        data_path=DATA_DEFAULT_PATH, data_subset=DataSubset.TRAIN
    )
    default_dataset_test = MoNuSegDataset(
        data_path=DATA_DEFAULT_PATH, data_subset=DataSubset.TEST
    )

    print("Parsing masks for training data")
    parse_masks(default_dataset_train)

    print("Parsing masks for testing data")
    parse_masks(default_dataset_test)

    print("Creating patches for training data")
    create_patches(default_dataset_train, DATA_PATCHES_PATH)

    print("Creating patches for testing data")
    create_patches(default_dataset_test, DATA_PATCHES_PATH)
