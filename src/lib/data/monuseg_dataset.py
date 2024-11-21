import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

from lib.util.file import (
    DATA_PATCHES_PATH,
    DEFAULT_MASK_FILE_EXTENSION,
    IMAGE_FILE_EXTENSION,
    PARSED_MASK_FILE_EXTENSION,
    PATCH_INFIX,
    default_mask_filename,
    image_filename,
    parsed_mask_filename,
)

from .data_subset import DataSubset


class MoNuSegDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        data_subset: DataSubset,
        color: bool = False,
        transform: Optional[A.BasicTransform] | Optional[A.BaseCompose] = None,
    ):
        self.data_path = data_path
        self.data_subset = data_subset
        self.color = color
        self.transform = transform

        self.identifiers = self.get_identifiers(data_subset)

    def get_identifiers(
        self,
        data_subset: Optional[DataSubset] = None,
        without_patch_suffix: bool = False,
    ) -> list[str]:
        if data_subset is None:
            data_subset = self.data_subset

        data_subset_path = self.data_path / data_subset.value
        if not data_subset_path.exists():
            raise FileNotFoundError(
                f"Data subset path {data_subset_path} does not exist"
            )

        image_identifiers = set()
        mask_identifiers = set()
        for candidate in data_subset_path.iterdir():
            if not candidate.is_file():
                continue

            if candidate.suffix == IMAGE_FILE_EXTENSION:
                image_identifier_candidate = candidate.stem

                if without_patch_suffix:
                    image_identifier_candidate = image_identifier_candidate.split(
                        PATCH_INFIX
                    )[0]

                image_identifiers.add(image_identifier_candidate)

            if candidate.suffix in {
                DEFAULT_MASK_FILE_EXTENSION,
                PARSED_MASK_FILE_EXTENSION,
            }:
                mask_identifier_candidate = candidate.stem.split("_")[0]

                if without_patch_suffix:
                    mask_identifier_candidate = mask_identifier_candidate.split(
                        PATCH_INFIX
                    )[0]

                mask_identifiers.add(mask_identifier_candidate)

        if image_identifiers != mask_identifiers:
            raise ValueError(
                f"Image and mask identifiers do not match for {data_subset.value}"
            )

        return sorted(list(image_identifiers))  # For deterministic order

    def load_image(self, identifier: str) -> torch.Tensor:
        image_path = (
            self.data_path / self.data_subset.value / image_filename(identifier)
        )
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist")

        return F.to_dtype(
            F.pil_to_tensor(Image.open(image_path)), dtype=torch.float32, scale=True
        )

    def load_mask(self, identifier: str, color: bool = False) -> torch.Tensor:
        color = self.color if self.color else color

        mask_path = (
            self.data_path
            / self.data_subset.value
            / parsed_mask_filename(identifier, color=color)
        )
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask file {mask_path} does not exist! Please run the preprocessing script first"
            )

        return torch.as_tensor(np.load(mask_path)).float()

    def load_mask_xml(self, identifier: str) -> ET.ElementTree:
        mask_path = (
            self.data_path / self.data_subset.value / default_mask_filename(identifier)
        )
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file {mask_path} does not exist")

        return ET.parse(mask_path)

    def get_patches_by_identifier(
        self, identifier: str
    ) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        if self.data_path != DATA_PATCHES_PATH:
            raise ValueError(
                "'get_patches_by_identifier' can only be called on patched data subsets"
            )

        patch_identifiers = map(
            lambda ident: str(ident).split(".")[0],
            (self.data_path / self.data_subset.value).glob(
                # Use the image file extension to avoid having to remove binary/color indication
                f"{identifier}*{IMAGE_FILE_EXTENSION}"
            ),
        )

        return [
            (
                self.load_image(patch_identifier),
                self.load_mask(patch_identifier),
                patch_identifier,
            )
            for patch_identifier in patch_identifiers
        ]

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        identifier = self.identifiers[idx]
        image = self.load_image(identifier)
        mask = self.load_mask(identifier)

        if self.transform:
            transformed = self.transform(
                image=image.numpy().transpose(1, 2, 0),
                mask=mask.numpy().transpose(1, 2, 0),
            )

            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask, identifier
