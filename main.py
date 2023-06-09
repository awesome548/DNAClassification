import torch
import pprint
from torch import nn
import os
import click
import numpy as np
from dotenv import load_dotenv
from model import effnetv2,EffNetV2
from ops_data.dataformat import Dataformat
from preference import model_preference,model_parameter
from torchmetrics import Accuracy,Recall,Precision,F1Score,ConfusionMatrix,AUROC
from ops_process import train_loop,test_loop
from torch.utils.tensorboard import SummaryWriter

@click.command()
@click.option('--arch', '-a', help='Name of Architecture')
@click.option('--batch', '-b', default=1000, help='Batch size, default 1000')
@click.option('--minepoch', '-e', default=10, help='Number of min epoches')
@click.option('--learningrate', '-lr', default=1e-2, help='Learning rate')
@click.option('--hidden', '-hidden', default=64, help='dim of hidden layer')
@click.option('--t_class', '-t', default=0, help='Target class index')
@click.option('--mode', '-m', default=0, help='0 : normal, 1: best')
@click.option('--classification', '-c', default="base", help='base, genus, family')

def main(arch, batch, minepoch, learningrate,hidden,t_class,mode,classification):
    load_dotenv()
    FAST5 = os.environ['FAST5']
    cutoff = int(os.environ['CUTOFF'])
    maxlen = int(os.environ['MAXLEN'])
    cutlen = int(os.environ['CUTLEN'])
    dataset_size = int(os.environ['DATASETSIZE'])
    writer = SummaryWriter('runs/experiment_1')
    load_model = False

    """
    Dataset preparation
    """
    ### Dataset  設定
    # 変更不可 .values()の取り出しあり
    cut_size = {
        'cutoff' : cutoff,
        'cutlen' : cutlen,
        'maxlen' : maxlen,
        'stride' : cutlen
    }
    pprint.pprint(cut_size,width=1)
    # fast5 -> 種のフォルダが入っているディレクトリ -> 対応の種のみを入れたディレクトリを使うこと！！
    # id list -> 種の名前に対応した.txtが入ったディレクトリ
    data = Dataformat(FAST5,dataset_size,cut_size,classification)
    train_loader,_, = data.loader(batch)
    test_loader = data.test_loader(batch)
    param = data.param()
    datasize,classes,ylabel = param['size'],param['num_cls'],param['ylabel']
    print(f'Num of Classes :{classes}')
    """
    Preference
    """
    # Model 設定
    # 変更不可 .values()の取り出しあり metrics.py
    pref = {
        "data_size" : datasize,
        "lr" : learningrate,
        "cutlen" : cutlen,
        "classes" : classes,
        "epoch" : minepoch,
        "target" : t_class,
        "name" : arch,
        "heatmap" : True,
        "y_label" : ylabel,
        "project" :  "Master_init",
    }
    pprint.pprint(pref,width=1)
    model,useModel = model_preference(arch,hidden,pref,mode=mode)
    """
    Training
    """
    ### Train ###
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # network, loss functions and optimizer
    # 変更不可 .values()の取り出しあり
    models = {
        "model" : model.to(device),
        "criterion" : nn.CrossEntropyLoss().to(device),
        "optimizer" : torch.optim.Adam(model.parameters(), lr=learningrate),
        "device" : device,
    }
    train_loop(models,pref,train_loader,load_model,writer)
    test_loop(models,pref, test_loader,load_model,writer)


if __name__ == '__main__':
    main()
