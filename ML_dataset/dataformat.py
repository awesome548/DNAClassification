import os
import glob
import pprint

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from ML_dataset.datamodule import DataModule
from ML_dataset.dataset import MultiDataset
from ML_preparation.preprocess import Preprocess
from ML_preparation.utils import calu_size

load_dotenv()
DATASIZE = int(os.environ["DATASETSIZE"])
FAST5 = os.environ["FAST5"]
CUTOFF = int(os.environ["CUTOFF"])
MAXLEN = int(os.environ["MAXLEN"])
CUTLEN = int(os.environ["CUTLEN"])
STRIDE = int(os.environ["STRIDE"])
NUM_WORKERS = min(os.cpu_count() or 4, 24)

## 変更不可 .values()の取り出しあり
CUTSIZE = {
    "cutoff": CUTOFF,
    "cutlen": CUTLEN,
    "maxlen": MAXLEN,
    "stride": STRIDE,
}


def _split_data(
    d_list: list[torch.Tensor], dataset_size: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Split each tensor in d_list into train/val/test (80/10/10)."""
    assert d_list[0].shape[0] == dataset_size
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size
    data_size = [train_size, val_size, test_size]
    train, val, test = [], [], []
    for data in d_list:
        tr, v, te = torch.split(data, data_size)
        train.append(tr)
        val.append(v)
        test.append(te)
    return train, val, test


def base_class(fast5_id: list) -> tuple:
    d_list = []
    for fast5, out, flag in fast5_id:
        pre = Preprocess(fast5, out, flag)
        d_list.append(pre.process())

    dataset_size = calu_size()
    train, val, test = _split_data(d_list, dataset_size)
    return train, val, test, dataset_size


def multi_class(fast5_id: list, ctgys: list) -> tuple:
    d_list = []
    l_list = []
    i = 0
    for fast5, out, flag in fast5_id:
        matched = False
        for ctgy in ctgys:
            if ctgy in fast5:
                pre = Preprocess(fast5, out, flag)
                d_list.append(pre.process())
                l_list.append(i)
                i += 1
                matched = True
                break
        if not matched:
            l_list.append(-1)

    dataset_size = calu_size()
    train, val, test = _split_data(d_list, dataset_size)
    return train, val, test, dataset_size, l_list


class Dataformat:
    def __init__(self, cls_type: str, use_category: tuple | None = None) -> None:
        fast5_set = []
        pprint.pprint(CUTSIZE, width=1)

        ## 種はターゲットディレクトリに種の名前のフォルダとfast5フォルダを作る
        if not os.path.exists(FAST5):
            raise FileNotFoundError(f"ディレクトリがありません: {FAST5}")

        # Directory starting with A-Z -> loaded : with "_" -> not loaded
        for name in glob.glob(FAST5 + "/[A-Z]*"):
            fast5_set.append([name, os.path.basename(name), False])

        ## ファイルの順番がわからなくなるためソート
        fast5_set.sort()

        ## 二値分類 / 多値分類 の場合わけ
        assert use_category != []
        if use_category:
            train, val, test, dataset_size, l_list = multi_class(fast5_set, use_category)
            cls_type = "multi_value"
        else:
            train, val, test, dataset_size = base_class(fast5_set)
            l_list = None

        self.training_set = MultiDataset(train, cls_type)
        self.validation_set = MultiDataset(val, cls_type)
        self.test_set = MultiDataset(test, cls_type)
        self.classes = self.training_set.num_classes

        captions = l_list if l_list else self.training_set.captions

        y_label: list[str | None] = [None] * self.classes
        for spe, cap in zip(fast5_set, captions):
            if cap >= 0:
                if y_label[cap] is None:
                    y_label[cap] = spe[1]
                else:
                    y_label[cap] += f"\n{spe[1]}"
        self.ylabel = y_label
        self.dataset = dataset_size

    def module(self, batch: int) -> DataModule:
        return DataModule(self.training_set, self.validation_set, self.test_set, batch_size=batch)

    def loader(self, batch: int) -> tuple[DataLoader, DataLoader]:
        params = {"batch_size": batch, "shuffle": True, "num_workers": NUM_WORKERS}
        return DataLoader(self.training_set, **params), DataLoader(self.validation_set, **params)

    def test_loader(self, batch: int) -> DataLoader:
        params = {"batch_size": batch, "shuffle": False, "num_workers": NUM_WORKERS}
        return DataLoader(self.test_set, **params)

    def param(self) -> dict:
        return {"size": self.dataset, "num_cls": self.classes, "ylabel": self.ylabel}
