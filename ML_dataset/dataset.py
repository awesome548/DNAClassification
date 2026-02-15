import torch

# 属 genus、科 family、目 order、綱 class、門 phylum、界 kingdom、超界 domain

# Taxonomy label mappings per classification mode
TAXONOMY_LABELS: dict[str, list[int]] = {
    "order":  [0, 1, 2, 3, 2, 2, 2, 3, 1, 0, 2, 1],
    "family": [0, 1, 2, 3, 2, 2, 2, 4, 5, 6, 2, 8],
    "class":  [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
}


def _concat_data(data: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(data)


def _build_labels(data: list[torch.Tensor], label: list[int]) -> torch.Tensor:
    parts = [
        torch.full((d.shape[0],), lbl, dtype=torch.int64)
        for d, lbl in zip(data, label)
    ]
    return torch.cat(parts)


def classification(mode: str, length: int) -> list[int]:
    if mode in TAXONOMY_LABELS:
        return TAXONOMY_LABELS[mode]
    return list(range(length))


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[torch.Tensor], mode: str) -> None:
        label = classification(mode, len(data))
        if len(label) != len(data):
            raise IndexError("label size does not match to dataset size")

        self.data = _concat_data(data)
        self.label = _build_labels(data, label)
        self.captions = label
        self.num_classes = max(label) + 1

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.label[index]
