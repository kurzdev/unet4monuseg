import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.augmentation import AugmentationClass, get_augmentation
from lib.data import DataSubset, MoNuSegDataset
from lib.metrics import dice_coeff
from lib.util.device import get_device
from lib.util.file import (
    DATA_DEFAULT_PATH,
    DATA_PATCHES_PATH,
    MODEL_METRICS_PATH,
    dice_scores_filename,
    losses_filename,
)
from lib.util.model import load_unet
from lib.util.plot import (
    plot_dice_score_boxplots,
    plot_dice_score_evolution,
    plot_dice_score_violin,
    plot_loss_evolution,
)
from lib.util.predict import predict_mask


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device | str,
) -> list[float]:
    dice_scores = list()
    for i in tqdm(range(len(dataset)), desc="Evaluating"):  # type: ignore
        image, true_mask, _ = dataset[i]

        dice_score = dice_coeff(
            predict_mask(model, image, device),
            true_mask.squeeze(),
        )
        dice_scores.append(dice_score.item())

    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Standard Deviation: {np.std(dice_scores):.4f}")

    return dice_scores


def load_metrics(timestamp: str) -> tuple[dict, dict]:
    metrics_path = MODEL_METRICS_PATH / timestamp

    with open(metrics_path / losses_filename(timestamp), "rb") as f:
        losses = pickle.load(f)

    with open(metrics_path / dice_scores_filename(timestamp), "rb") as f:
        dice_scores = pickle.load(f)

    return losses, dice_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate UNet model metrics")
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp of the model to evaluate",
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
        help="Epoch of the model used to evaluate",
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

    data_path = DATA_PATCHES_PATH if not args.fullsize else DATA_DEFAULT_PATH
    transforms = (
        get_augmentation(
            [
                AugmentationClass(augmentation_class)
                for augmentation_class in args.augmentation
            ]
        )
        if args.augmentation
        else None
    )

    train_dataset = MoNuSegDataset(
        data_path=data_path,
        data_subset=DataSubset.TRAIN,
        transform=transforms,
    )
    test_dataset = MoNuSegDataset(
        data_path=data_path,
        data_subset=DataSubset.TEST,
        transform=transforms,
    )

    model = load_unet(args.timestamp, args.epoch, device, args.num_filters, args.depth)

    losses, dice_scores = load_metrics(args.timestamp)
    plot_loss_evolution(losses)
    plot_dice_score_evolution(dice_scores)

    print("Evaluating model on train dataset...")
    dice_scores_train = evaluate_model(model, train_dataset, device)

    print("Evaluating model on test dataset...")
    dice_scores_test = evaluate_model(model, test_dataset, device)

    plot_dice_score_boxplots({"Train": dice_scores_train, "Test": dice_scores_test})
    plot_dice_score_violin({"Train": dice_scores_train, "Test": dice_scores_test})
