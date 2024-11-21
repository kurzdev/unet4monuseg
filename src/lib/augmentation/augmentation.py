import albumentations as A
from albumentations.pytorch import ToTensorV2

from .augmentation_class import AugmentationClass


def get_augmentation(augmentation_classes: list[AugmentationClass]) -> A.BaseCompose:
    augmentations = []

    if AugmentationClass.ARTEFACT in augmentation_classes:
        augmentations.extend(
            [
                A.GaussNoise(p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
            ]
        )

    if AugmentationClass.AFFINE_FLIP in augmentation_classes:
        augmentations.extend(
            [
                A.Affine(translate_percent=(0.1, 0.1), rotate=0, p=0.5),
                A.Affine(rotate=30, p=0.5),
                A.Affine(scale=(0.8, 1.2), p=0.5),
                A.HorizontalFlip(p=0.5),
            ]
        )

    return A.Compose(augmentations + [ToTensorV2(transpose_mask=True)])
