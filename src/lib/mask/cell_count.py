import torch


def dfs(mask: torch.Tensor) -> int:
    def dfs_visit(i: int, j: int) -> None:
        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            return

        if mask[i, j] == 0:
            return

        mask[i, j] = 0

        dfs_visit(i - 1, j)
        dfs_visit(i + 1, j)
        dfs_visit(i, j - 1)
        dfs_visit(i, j + 1)

    count = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                dfs_visit(i, j)
                count += 1

    return count


def union_find(mask: torch.Tensor) -> int:
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])

        return parent[x]

    def union(x: int, y: int) -> None:
        root_x = find(x)
        root_y = find(y)

        if root_x != root_y:
            parent[root_x] = root_y

    parent = [i for i in range(mask.shape[0] * mask.shape[1])]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                continue

            if i > 0 and mask[i - 1, j] == 1:
                union(i * mask.shape[1] + j, (i - 1) * mask.shape[1] + j)

            if j > 0 and mask[i, j - 1] == 1:
                union(i * mask.shape[1] + j, i * mask.shape[1] + j - 1)

    count = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (
                mask[i, j] == 1
                and parent[i * mask.shape[1] + j] == i * mask.shape[1] + j
            ):
                count += 1

    return count
