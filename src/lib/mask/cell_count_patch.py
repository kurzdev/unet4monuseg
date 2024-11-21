import torch
from skimage.measure import label, regionprops


def dfs_patch(mask: torch.Tensor) -> int:
    def dfs_visit(i: int, j: int, component_pixels: set) -> None:
        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            return

        if mask[i, j] == 0:
            return

        mask[i, j] = 0
        component_pixels.add((i, j))

        # Explore neighbors in the 4 directions
        dfs_visit(i - 1, j, component_pixels)
        dfs_visit(i + 1, j, component_pixels)
        dfs_visit(i, j - 1, component_pixels)
        dfs_visit(i, j + 1, component_pixels)

    components = []

    # Find all connected components and categorize them
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                component_pixels = set()
                dfs_visit(i, j, component_pixels)

                # Check if the component touches any boundary
                touches_boundary = any(
                    i == 0 or i == mask.shape[0] - 1 or j == 0 or j == mask.shape[1] - 1
                    for i, j in component_pixels
                )

                components.append((len(component_pixels), touches_boundary))

    return _compute_component_count(components)


def skimage_patch(mask: torch.Tensor) -> int:
    mask = mask.numpy()  # type: ignore

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    components = []

    for region in regions:
        bbox = region.bbox
        touches_boundary = (
            bbox[0] == 0
            or bbox[1] == 0
            or bbox[2] == mask.shape[0]
            or bbox[3] == mask.shape[1]
        )

        components.append((region.area, touches_boundary))

    return _compute_component_count(components)


def _compute_component_count(components: list[tuple[int, bool]]) -> int:
    count = 0

    # Expecting a list of tuples (size, touches_boundary)
    non_boundary_sizes = [comp[0] for comp in components if not comp[1]]

    # Compute the average size of non-boundary components
    avg_size_non_boundary = (
        sum(non_boundary_sizes) / len(non_boundary_sizes) if non_boundary_sizes else 0
    )

    # Now, count valid components
    for component in components:
        if not component[1] or (component[0] > (0.5 * avg_size_non_boundary)):
            count += 1

    return count
