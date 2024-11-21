import sys
from typing import Optional

import torch
import torch.nn as nn

from lib.model import UNet
from lib.util.device import get_device
from lib.util.file import (
    state_dict_filename,
    state_dict_path,
)


def load_unet(
    timestamp: str,
    epoch: Optional[int] = None,
    device: Optional[torch.device] = None,
    n_filters: int = 32,
    depth: int = 4,
) -> nn.Module:
    if device is None:
        device = get_device()

    model = UNet(in_channels=3, out_channels=1, n_filters=n_filters, depth=depth)

    try:
        model.load_state_dict(
            torch.load(
                state_dict_path(timestamp, epoch)
                / state_dict_filename(timestamp, epoch),
                weights_only=True,
            )["model"]
        )
    except FileNotFoundError:
        print("Model could not be loaded. Please check the timestamp and epoch.")
        sys.exit(1)

    model = model.to(device)

    return model
