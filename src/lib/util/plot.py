import math
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_single_predicted_mask(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    pred_mask: torch.Tensor,
) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for ax in axs.ravel():
        ax.set_axis_off()

    axs[0].imshow(image.cpu().numpy().transpose(1, 2, 0))
    axs[0].set_title("Image")

    true_mask = ground_truth.cpu().numpy().squeeze()
    axs[1].imshow(true_mask, cmap="gray")
    axs[1].set_title("Ground Truth")

    model_mask = pred_mask.cpu().detach().numpy().squeeze() > 0.5
    axs[2].imshow(model_mask, cmap="gray")
    axs[2].set_title("Predicted Mask")

    axs[3].imshow(abs(true_mask - model_mask), cmap="gray")
    axs[3].set_title("Difference")

    plt.show()


def plot_predicted_mask_evolution(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    pred_masks: list[tuple[int, torch.Tensor]],
) -> None:
    rows = 1 + math.ceil(len(pred_masks) / 4)
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for ax in axs.ravel():
        ax.set_axis_off()

    axs[0, 0].imshow(image.cpu().numpy().transpose(1, 2, 0))
    axs[0, 0].set_title("Image")

    axs[0, 1].imshow(ground_truth.cpu().numpy().squeeze(), cmap="gray")
    axs[0, 1].set_title("Ground Truth")

    for i, (epoch, pred_mask) in enumerate(pred_masks):
        row = 1 + i // 4
        col = i % 4

        axs[row, col].imshow(
            (pred_mask.cpu().detach().numpy().squeeze() > 0.5), cmap="gray"
        )
        axs[row, col].set_title(f"Epoch {epoch}")

    plt.show()


def plot_dice_score_boxplots(dice_scores: dict) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    values = list(dice_scores.values())
    labels = list(dice_scores.keys())

    plt.boxplot(values)

    ax.set_xticklabels(labels)
    plt.xlabel("Dataset")
    plt.ylabel("Dice Score")
    plt.title("Dice Scores across Evaluation Scenarios")

    plt.show()


def plot_dice_score_violin(dice_scores: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    values = list(dice_scores.values())
    labels = list(dice_scores.keys())

    ax.violinplot(values, showmeans=True)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    plt.xlabel("Dataset")
    plt.ylabel("Dice Score")
    plt.title("Dice Scores across Evaluation Scenarios")

    plt.show()


def plot_dice_score_evolution(dice_scores: dict) -> None:
    _multi_line_plot(
        {
            "Training Dice Score": dice_scores["train"],
            "Validation Dice Score": dice_scores["val"],
        },
        "Training and Validation Dice Score",
        "Epoch",
        "Dice Score",
        annotate_extremum="max",
    )


def plot_loss_evolution(losses: dict) -> None:
    _multi_line_plot(
        {"Training Loss": losses["train"], "Validation Loss": losses["val"]},
        "Training and Validation Loss",
        "Epoch",
        "Loss",
        annotate_extremum="min",
    )


def _multi_line_plot(
    data: dict,
    title: str,
    x_label: str,
    y_label: str,
    annotate_extremum: Optional[Literal["min", "max"]] = None,
) -> None:
    plt.figure(figsize=(10, 5))

    for key, values in data.items():
        plt.plot(values, label=key)

        if annotate_extremum:
            extremum = getattr(np, f"arg{annotate_extremum}")(values)
            plt.annotate(
                f"{values[extremum]:.4f}",
                (extremum, values[extremum]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                arrowprops=dict(arrowstyle="->"),
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()
    plt.show()
