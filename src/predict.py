import argparse
from pathlib import Path

from lib.augmentation import AugmentationClass, get_augmentation
from lib.data import DataSubset, MoNuSegDataset
from lib.util.device import get_device
from lib.util.file import (
    DATA_DEFAULT_PATH,
    DATA_PATCHES_PATH,
    STATE_DICT_FILE_EXTENSION,
    state_dict_path,
)
from lib.util.model import load_unet
from lib.util.plot import plot_predicted_mask_evolution, plot_single_predicted_mask
from lib.util.predict import predict_mask


def evenly_sample_checkpoints(checkpoints: list[Path]) -> list[Path]:
    max_checkpoints = 4

    if len(checkpoints) <= max_checkpoints:
        return checkpoints

    step = len(checkpoints) // max_checkpoints
    sampled_checkpoints = checkpoints[::step]

    return sampled_checkpoints[:max_checkpoints]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Make predictions using UNet model")
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp of the model used to predict",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        required=False,
        default=32,
        help="Number of filters in the first layer of the UNet",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=False,
        default=4,
        help="Depth of the UNet",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=False,
        default=None,
        help="Epoch of the model used to predict",
    )
    parser.add_argument(
        "--evolution",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Plot evolution of predicted masks",
    )
    parser.add_argument(
        "--augmentation",
        nargs="+",
        type=str,
        choices=[augmentation_class.value for augmentation_class in AugmentationClass],
        required=False,
        default=None,
        help="Augmentation classes to apply to images and masks before prediction",
    )
    parser.add_argument(
        "--fullsize",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Use full size images and masks",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dataset = MoNuSegDataset(
        data_path=DATA_PATCHES_PATH if not args.fullsize else DATA_DEFAULT_PATH,
        data_subset=DataSubset.TEST,
        transform=get_augmentation(
            [
                AugmentationClass(augmentation_class)
                for augmentation_class in args.augmentation
            ]
        )
        if args.augmentation
        else None,
    )

    image, ground_truth, _ = dataset[0]

    if args.evolution:
        checkpoints = sorted(
            # Pass 1 as a dummy epoch to get all checkpoints
            state_dict_path(args.timestamp, 1).glob(f"*{STATE_DICT_FILE_EXTENSION}"),
            key=lambda x: x.stem,
        )
        sampled_checkpoints = evenly_sample_checkpoints(checkpoints)
        epoch_checkpoints = [
            (int(checkpoint.stem.split("_")[-1]), checkpoint)
            for checkpoint in sampled_checkpoints
        ]

        pred_masks = []
        for epoch, checkpoint in epoch_checkpoints:
            model = load_unet(
                args.timestamp, epoch, device, args.num_filters, args.depth
            )

            pred_mask = predict_mask(model, image, device) > 0.5
            pred_masks.append((epoch, pred_mask))

        plot_predicted_mask_evolution(image, ground_truth, pred_masks)

    model = load_unet(args.timestamp, args.epoch, device, args.num_filters, args.depth)
    print(
        f"Loaded model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters..."
    )

    pred_mask = predict_mask(model, image, device)
    plot_single_predicted_mask(image, ground_truth, pred_mask)
