import torch
from torch import nn
import os
import click
import datetime
import math
import numpy as np
from dotenv import load_dotenv
import pytorch_lightning as pl
from model import effnetv2,EffNetV2
from ops_process import evaluation
from ops_data.dataformat import Dataformat
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from preference import model_preference,model_parameter,logger_preference
from torchmetrics import Accuracy,Recall,Precision,F1Score,ConfusionMatrix,AUROC
import tqdm


### TRAIN and TEST ###
def train_loop(model, device, train_loader, criterion,optimizer,epoch) -> None:
    model.train()
    train_loss = 0
    for data, target in tqdm.tqdm(train_loader,leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    cur_loss = train_loss / len(train_loader)
    print('| epoch {:3d} | loss {:5.2f} | ppl {:8.2f}'.format(epoch,cur_loss, math.exp(cur_loss)))
    train_loss = 0

def test_loop(model, device, test_loader,criterion,n_class,t_class,heatmap):
    model.eval()
    with torch.no_grad():
        labels = torch.zeros(1)
        outputs = torch.zeros(1,n_class)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x,text="test")
            loss = criterion(y_hat,y)
            labels = torch.hstack((labels,y.clone().detach().cpu()))
            outputs = torch.vstack((outputs,y_hat.clone().detach().cpu()))

    outputs = outputs[1:,]
    labels = labels[1:]
    hidd_vec = model.cluster[1:]
    pref = model.pref
    y_hat_idx = outputs.max(dim=1).indices
    y_hat_idx = (y_hat_idx == t_class)
    y = (labels == t_class)

    evaluation(y_hat_idx,y_hat,y,n_class,t_class,hidd_vec,labels,pref)

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
    print(f'dataset size : {dataset_size}')
    print(f'Cutlen : {cut_size}')
    # fast5 -> 種のフォルダが入っているディレクトリ -> 対応の種のみを入れたディレクトリを使うこと！！
    # id list -> 種の名前に対応した.txtが入ったディレクトリ
    data = Dataformat(IDLIST,FAST5,dataset_size,cut_size)
    train_loader,val_dataloader,test_loader = data.loader(batch)
    dataset_size = data.size()
    classes = len(data)
    """
    Preference
    """
    project_name = "Master_init"
    heatmap = True
    train = False
    cfgs =[
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  6, 1, 1],
        [6, 256,  6, 2, 1],
    ]
    # Model 設定
    model,useModel = model_preference(arch,hidden,classes,cutlen,learningrate,t_class,minepoch,heatmap,project_name,mode=mode)
    """
    Training
    """
    ### Train ###
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # network, loss functions and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    if train:
        print("#######Train Start...")
        for epoch in (range(minepoch)):
            train_loop(model, device, train_loader, criterion,optimizer,epoch)
        torch.save(model, f'{MODEL}/{arch}-{datetime.date.today()}.pth')
    else:
        model = torch.load(f'{MODEL}/{arch}-{datetime.date.today()}.pth')

    model.eval()
    # testing with validation data
    print("#######Test Start...")
    test_loop(model, device, test_loader,criterion,classes,t_class,heatmap)


if __name__ == '__main__':
    main()
