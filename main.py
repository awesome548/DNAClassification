import torch
import pprint
from torch import nn
import os
import click
import numpy as np
from dotenv import load_dotenv
import pytorch_lightning as pl
from model import effnetv2,EffNetV2
from ops_data.dataformat import Dataformat
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from preference import model_preference,model_parameter
from torchmetrics import Accuracy,Recall,Precision,F1Score,ConfusionMatrix,AUROC
from ops_process import train_loop,test_loop
from torch.utils.tensorboard import SummaryWriter

@click.command()
@click.option('--arch', '-a', help='Name of Architecture')
@click.option('--batch', '-b', default=1000, help='Batch size, default 1000')
@click.option('--minepoch', '-me', default=10, help='Number of min epoches')
@click.option('--learningrate', '-lr', default=1e-2, help='Learning rate')
@click.option('--hidden', '-hidden', default=64, help='dim of hidden layer')
@click.option('--t_class', '-t', default=0, help='Target class index')
@click.option('--mode', '-m', default=0, help='0 : normal, 1: best')

def main(arch, batch, minepoch, learningrate,hidden,t_class,mode):
    load_dotenv()
    IDLIST = os.environ['IDLIST']
    FAST5 = os.environ['FAST5']
    MODEL = os.environ['MODEL']
    cutoff = int(os.environ['CUTOFF'])
    maxlen = int(os.environ['MAXLEN'])
    cutlen = int(os.environ['CUTLEN'])
    dataset_size = int(os.environ['DATASETSIZE'])
    writer = SummaryWriter('runs/experiment_1')

    ## 結果を同じにする
    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True
    #torch.set_deterministic_debug_mode(True)
    """
    Dataset preparation
    """
    ### Dataset  設定
    cut_size = {
        'cutoff' : cutoff,
        'cutlen' : cutlen,
        'maxlen' : maxlen,
        'stride' : cutlen
    }
    pprint.pprint(cut_size,width=1)
    # fast5 -> 種のフォルダが入っているディレクトリ -> 対応の種のみを入れたディレクトリを使うこと！！
    # id list -> 種の名前に対応した.txtが入ったディレクトリ
    data = Dataformat(IDLIST,FAST5,dataset_size,cut_size)
    train_loader,val_dataloader,test_loader = data.loader(batch)
    dataset_size = data.size()
    classes = len(data)
    print(f'Num of Classes :{classes}')
    """
    Preference
    """
    project_name = "Master_init"
    heatmap = True
    load_model = False
    # Model 設定
    pref = {
        "lr" : learningrate,
        "cutlen" : cutlen,
        "classes" : classes,
        "epoch" : minepoch,
        "target" : t_class,
        "name" : arch,
        "heatmap" : heatmap,
        "project" : project_name,
    }
    pprint.pprint(pref,width=1)
    model,useModel = model_preference(arch,hidden,pref,mode=mode)
    """
    Training
    """
    ### Train ###
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # network, loss functions and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
    
    train_loop(model,device,train_loader,criterion,optimizer,minepoch,load_model,arch,writer)
    test_loop(model, device, test_loader,criterion,classes,t_class,load_model,writer)


if __name__ == '__main__':
    main()
