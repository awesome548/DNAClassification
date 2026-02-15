import os
import pprint

import click
import torch
from dotenv import load_dotenv
from torch import nn

from ML_dataset import Dataformat
from ML_model import model_preference
from ML_processing import train_loop


@click.command()
@click.option("--arch", "-a", required=True, help="Name of Architecture")
@click.option("--batch", "-b", default=1000, help="Batch size, default 1000")
@click.option("--minepoch", "-e", default=20, help="Number of min epochs")
@click.option("--learningrate", "-lr", default=1e-2, help="Learning rate")
@click.option("--hidden", "-hidden", default=64, help="dim of hidden layer")
@click.option("--t_class", "-t", default=0, help="Target class index")
@click.option("--mode", "-m", default=0, help="0 : normal, 1: best")
@click.option("--cls_type", "-c", default="base", help="base, genus, family")
@click.option("--category1", "-c1", default="n", help="Category filter 1")
@click.option("--category2", "-c2", default="n", help="Category filter 2")
def main(
    arch: str,
    batch: int,
    minepoch: int,
    learningrate: float,
    hidden: int,
    t_class: int,
    mode: int,
    cls_type: str,
    category1: str,
    category2: str,
) -> None:
    load_dotenv()
    cutlen = int(os.environ["CUTLEN"])
    load_model = False

    use_category = (category1, category2) if category1 != "n" else None

    # --- Dataset preparation / データセット設定 ---
    data = Dataformat(cls_type, use_category)
    train_loader, val_loader = data.loader(batch)
    test_loader = data.test_loader(batch)
    param = data.param()
    datasize, classes, ylabel = param["size"], param["num_cls"], param["ylabel"]
    print(f"Num of Classes: {classes}")

    # --- Model preference / Model設定 ---
    ## 変更不可 .values()の取り出しあり metrics.py
    pref = {
        "data_size": datasize,
        "lr": learningrate,
        "cutlen": cutlen,
        "classes": classes,
        "epoch": minepoch,
        "target": t_class,
        "name": arch,
        "confmat": False,
        "heatmap": False,
        "y_label": ylabel,
        "project": "gigascience",
        "category": cls_type,
    }
    pprint.pprint(pref, width=1)
    model, _ = model_preference(arch, hidden, pref, mode=mode)

    # --- Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 変更不可 .values()の取り出しあり loop.py
    models = {
        "model": model.to(device),
        "criterion": nn.CrossEntropyLoss().to(device),
        "optimizer": torch.optim.Adam(model.parameters(), lr=learningrate),
        "device": device,
    }
    train_loop(models, pref, train_loader, val_loader, load_model)


if __name__ == "__main__":
    main()
