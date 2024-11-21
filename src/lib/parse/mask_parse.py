import xml.etree.ElementTree as ET

import matplotlib.path
import torch
from tqdm import tqdm


def parse_mask_xml(
    mask_xml: ET.ElementTree, nrow: int, ncol: int
) -> tuple[torch.Tensor, torch.Tensor]:
    root = mask_xml.getroot()
    regions = root.findall(".//Region")

    binary_mask = torch.zeros((1, nrow, ncol), dtype=torch.bool)
    color_mask = torch.zeros((3, nrow, ncol), dtype=torch.float32)

    for region in tqdm(regions, desc="Parsing regions for mask"):
        vertices = region.findall(".//Vertex")
        xy = torch.tensor(
            [
                (float(vertex.get("X")), float(vertex.get("Y")))  # type: ignore
                for vertex in vertices
            ]
        )

        path = matplotlib.path.Path(xy.numpy())

        grid_x, grid_y = torch.meshgrid(
            torch.arange(ncol), torch.arange(nrow), indexing="xy"
        )
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        mask = torch.tensor(path.contains_points(grid_points.numpy())).reshape(
            (nrow, ncol)
        )

        binary_mask[0, mask] = 1
        color_mask[:, mask] = torch.rand(3).unsqueeze(-1)

    return binary_mask, color_mask
