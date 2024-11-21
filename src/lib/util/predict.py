import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_mask(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    model.eval()

    with torch.inference_mode():
        image = (
            image.unsqueeze(0)
            .to(device=device, memory_format=torch.channels_last)
            .contiguous()
        )
        pred_mask = model(image)

    return F.sigmoid(pred_mask.cpu().detach().squeeze())
