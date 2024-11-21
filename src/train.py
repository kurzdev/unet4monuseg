import argparse
import pickle
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from lib.augmentation import AugmentationClass, get_augmentation
from lib.data import DataSubset, MoNuSegDataset
from lib.metrics import dice_coeff, dice_loss
from lib.model import UNet
from lib.util.device import get_device
from lib.util.early_stopping import EarlyStopping
from lib.util.file import (
    DATA_PATCHES_PATH,
    MODEL_METRICS_PATH,
    MODELS_PATH,
    dice_scores_filename,
    losses_filename,
    state_dict_filename,
    state_dict_path,
)

# Implementation based on https://github.com/milesial/Pytorch-UNet/tree/master


def train_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device | str,
    epochs: int = 300,
    batch_size: int = 8,
    learning_rate: float = 1e-2,
    momentum: float = 0.99,
    weight_decay: float = 3e-5,
    val_percent: float = 0.2,
    use_early_stopping: bool = True,
    checkpoint: Optional[dict] = None,
) -> None:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_epoch = 0
    print(f"Starting training at {current_time} for {epochs} epochs")

    num_val = int(len(dataset) * val_percent)  # type: ignore
    num_train = len(dataset) - num_val  # type: ignore
    print(f"Training on {num_train} samples, validating on {num_val} samples")

    train_dataset, val_dataset = random_split(
        dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    # According to https://www.nature.com/articles/s41592-020-01008-z
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
        foreach=True,
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, epochs, 0.9)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    early_stopping = EarlyStopping()

    losses = {"train": list(), "val": list()}
    dice_scores = {"train": list(), "val": list()}

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch"):
        ### Training ###
        model.train()

        train_loss = 0
        train_dice_score = 0

        for batch in tqdm(train_loader, desc="Train", leave=False):
            images, true_masks, _ = batch

            assert images.shape[1] == model.in_channels, (
                f"Network has been defined with {model.in_channels} input channels, "
                f"but loaded images have {images.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )

            images = images.to(
                device=device, memory_format=torch.channels_last
            ).contiguous()
            true_masks = true_masks.to(
                device=device, memory_format=torch.channels_last
            ).contiguous()

            pred_masks = model(images)

            loss, dice_score = _calculate_loss_dice_score(
                pred_masks, true_masks, criterion
            )
            train_loss += loss.item()
            train_dice_score += dice_score.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_dice_score /= len(train_loader)
        print(f"Mean Training Loss: {train_loss:.4f}")
        print(f"Mean Training Dice Score: {train_dice_score:.4f}")

        losses["train"].append(train_loss)
        dice_scores["train"].append(train_dice_score)

        ### Validation ###
        model.eval()

        with torch.no_grad():
            val_loss = 0
            val_dice_score = 0

            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images, true_masks, _ = batch

                images = images.to(
                    device=device,
                    memory_format=torch.channels_last,
                ).contiguous()
                true_masks = true_masks.to(
                    device=device,
                    memory_format=torch.channels_last,
                ).contiguous()

                pred_masks = model(images)

                loss, dice_score = _calculate_loss_dice_score(
                    pred_masks, true_masks, criterion
                )
                val_loss += loss.item()
                val_dice_score += dice_score.item()

            val_loss /= len(val_loader)
            val_dice_score /= len(val_loader)
            print(f"Mean Validation Loss: {val_loss:.4f}")
            print(f"Mean Validation Dice Score: {val_dice_score:.4f}")

            if use_early_stopping and epoch > 50 and early_stopping(val_loss):
                # We want the model to train for at least 50 epochs before stopping to cut off early fluctuations
                print("Early stopping triggered!")
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Detected improved validation loss! Saving best model")

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }

                checkpoint_path = state_dict_path(current_time, epoch)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint,
                    checkpoint_path / state_dict_filename(current_time, epoch),
                )

            losses["val"].append(val_loss)
            dice_scores["val"].append(val_dice_score)

            scheduler.step()

    print("Training finished!")
    torch.save(model.state_dict(), MODELS_PATH / state_dict_filename(current_time))

    # Small man's tensorboard
    metrics_path = MODEL_METRICS_PATH / f"{current_time}"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_path / losses_filename(current_time), "wb") as f:
        pickle.dump(losses, f)

    with open(metrics_path / dice_scores_filename(current_time), "wb") as f:
        pickle.dump(dice_scores, f)


def _calculate_loss_dice_score(
    pred_masks: torch.Tensor, true_masks: torch.Tensor, criterion: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    loss = criterion(pred_masks.squeeze(1), true_masks.squeeze(1))
    loss += dice_loss(F.sigmoid(pred_masks.squeeze(1)), true_masks.squeeze(1))
    loss /= 2

    dice_score = dice_coeff(
        F.sigmoid(pred_masks.squeeze(1)),
        true_masks.squeeze(1),
        reduce_batch_first=True,
    )

    return loss, dice_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train UNet model")
    parser.add_argument(
        "--augmentation",
        nargs="+",
        type=str,
        choices=[augmentation_class.value for augmentation_class in AugmentationClass],
        required=False,
        default=None,
        help="Augmentation classes to apply to images and masks before training",
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
        "--epochs",
        type=int,
        required=False,
        default=300,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--from_checkpoint",
        nargs=2,
        type=str,
        required=False,
        default=None,
        help="Resume training from $timestamp, $epoch",
    )

    args = parser.parse_args()

    model = UNet(
        in_channels=3, out_channels=1, n_filters=args.num_filters, depth=args.depth
    )

    dataset = MoNuSegDataset(
        data_path=DATA_PATCHES_PATH,
        data_subset=DataSubset.TRAIN,
        transform=get_augmentation(
            [
                AugmentationClass(augmentation_class)
                for augmentation_class in args.augmentation
            ]
        )
        if args.augmentation
        else None,
    )

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = None
    if args.from_checkpoint:
        timestamp, epoch = args.from_checkpoint
        checkpoint = torch.load(
            state_dict_path(timestamp, int(epoch))
            / state_dict_filename(timestamp, int(epoch))
        )

    train_model(
        model=model,
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        checkpoint=checkpoint,
    )
