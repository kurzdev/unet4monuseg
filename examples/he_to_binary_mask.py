import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from PIL import Image
from tqdm import tqdm


def he_to_binary_mask(filename):
    # Load image
    im_file = f"{filename}.tif"
    image = Image.open(im_file)
    ncol, nrow = image.size  # Image width, height

    # Read XML file
    xml_file = f"{filename}.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize masks
    binary_mask = np.zeros((nrow, ncol), dtype=int)
    color_mask = np.zeros((nrow, ncol, 3), dtype=float)

    regions = root.findall(".//Region")  # Get all 'Region' elements

    # Process each region
    for zz, region in enumerate(tqdm(regions), 1):
        # Extract vertices for this region
        vertices = region.findall(".//Vertex")
        xy = np.array(
            [(float(vertex.get("X")), float(vertex.get("Y"))) for vertex in vertices]
        )

        # Create a polygon from vertices
        path = Path(xy)
        grid_x, grid_y = np.meshgrid(np.arange(ncol), np.arange(nrow))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        mask = path.contains_points(grid_points).reshape((nrow, ncol))

        # Update binary mask
        binary_mask += zz * (1 - np.minimum(1, binary_mask)) * mask

        # Assign random colors for the color mask
        color_mask += (
            np.random.rand(1, 3) * mask[..., np.newaxis]
        )  # Adding random colors

    # Convert binary mask to pure 0 or 1 values
    binary_mask = np.clip(binary_mask, 0, 1)

    # Show binary mask and color mask
    plt.figure()
    plt.imshow(binary_mask)
    plt.title("Binary Mask")

    plt.figure()
    plt.imshow(color_mask)
    plt.title("Color Mask")
    plt.show()

    return binary_mask, color_mask


# Example usage
filename = "data/train/TCGA-CH-5767-01Z-00-DX1"  # Replace with your actual filename (without extension)
binary_mask, color_mask = he_to_binary_mask(filename)
