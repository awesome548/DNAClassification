import glob
import os

import torch
from torch.utils.data import DataLoader

from ML_preparation.preprocess import Preprocess
from ML_preparation.utils import calu_size

NUM_WORKERS = min(os.cpu_count() or 4, 24)


def base_class(
    idset: list[str], dataset: list[str], size: int, cut_size: dict
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int]:
    all_data = [
        Preprocess(idset[i]).process(inpath=dataset[i], **cut_size, size=size)
        for i in range(len(idset))
    ]

    cutlen, maxlen, stride = cut_size["cutlen"], cut_size["maxlen"], cut_size["stride"]
    manipulate = calu_size(cutlen, maxlen, stride)
    dataset_size = manipulate * size
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size
    data_size = [train_size, val_size, test_size]

    train, val, test = [], [], []
    for d in all_data:
        tr, v, te = torch.split(d, data_size)
        train.append(tr)
        val.append(v)
        test.append(te)

    return train, val, test, dataset_size


class Dataformat2:
    def __init__(
        self,
        target: str,
        inpath: str,
        dataset_size: int,
        cut_size: dict,
        num_classes: int,
        idx: torch.Tensor,
    ) -> None:
        idset = sorted(glob.glob(target + "/*.txt"))
        dataset = sorted(glob.glob(inpath + "/*"))

        _, _, test, _ = base_class(idset, dataset, dataset_size, cut_size)
        self.test_set = MultiDataset2(test, num_classes, idx)

    def test_loader(self, batch: int) -> DataLoader:
        params = {"batch_size": batch, "shuffle": False, "num_workers": NUM_WORKERS}
        return DataLoader(self.test_set, **params)


def _category_data(data: list[torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    return torch.cat(data)[idx.cpu(), :]


def _category_label(data: list[torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    """Build binary labels: first species = 1, second = 0, rest = 1."""
    idx_cpu = idx.cpu()
    labels = [
        torch.zeros(d.shape[0]) if i == 1 else torch.ones(d.shape[0])
        for i, d in enumerate(data)
    ]
    return torch.cat(labels, dim=0)[idx_cpu].to(torch.int64)


class MultiDataset2(torch.utils.data.Dataset):
    def __init__(self, data: list[torch.Tensor], num_classes: int, idx: torch.Tensor) -> None:
        self.data = _category_data(data, idx)
        self.label = _category_label(data, idx)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.label[index]
