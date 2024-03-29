import os
import pprint
import click
import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ML_dataset import Dataformat
from ML_processing import test_loop, train_loop
from ML_model import model_preference

@click.command()
@click.option("--arch", "-a", help="Name of Architecture")
@click.option("--batch", "-b", default=1000, help="Batch size, default 1000")
@click.option("--minepoch", "-e", default=20, help="Number of min epoches")
@click.option("--learningrate", "-lr", default=1e-2, help="Learning rate")
@click.option("--hidden", "-hidden", default=64, help="dim of hidden layer")
@click.option("--t_class", "-t", default=0, help="Target class index")
@click.option("--mode", "-m", default=0, help="0 : normal, 1: best")
@click.option("--cls_type", "-c", default="base", help="base, genus, family")
@click.option("--category1", "-c1", default="n", help="")
@click.option("--category2", "-c2", default='n', help="")

def main(arch, batch, minepoch, learningrate, hidden, t_class, mode, cls_type,category1,category2):
    load_dotenv()
    cutlen = int(os.environ["CUTLEN"])
    writer = SummaryWriter("runs/accuracy")
    load_model = False
    
    if category1 != "n":
        use_category = (category1,category2)
    else:
        use_category = None

    """
    Dataset preparation
    データセット設定
    """
    ## fast5 -> 種のフォルダが入っているディレクトリ -> 対応の種のみを入れたディレクトリを使うこと！！
    ## id list -> 種の名前に対応した.txtが入ったディレクトリ
    data = Dataformat(cls_type,use_category)
    train_loader, _ = data.loader(batch)
    test_loader = data.test_loader(batch)
    param = data.param()
    datasize, classes, ylabel = param["size"], param["num_cls"], param["ylabel"]
    print(f"Num of Classes :{classes}")
    """
    Preference
    Model設定
    """
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
        "category" : cls_type,
    }
    pprint.pprint(pref, width=1)
    model, useModel = model_preference(arch, hidden, pref, mode=mode)
    """
    Training
    """
    ### Train ###
    if torch.cuda.is_available:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### network, loss functions and optimizer
    ## 変更不可 .values()の取り出しあり loop.py
    models = {
        "model": model.to(device),
        "criterion": nn.CrossEntropyLoss().to(device),
        "optimizer": torch.optim.Adam(model.parameters(), lr=learningrate),
        "device": device,
    }
    train_loop(models, pref, train_loader, load_model, writer)
    test_loop(models, pref, test_loader, load_model, writer,use_category)


if __name__ == "__main__":
    main()
