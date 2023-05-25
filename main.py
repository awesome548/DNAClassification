import torch
from torch import nn
import os
import click
import time
import math
import numpy as np
from dotenv import load_dotenv
import pytorch_lightning as pl
from model import effnetv2,EffNetV2
from ops_data.dataformat import Dataformat
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from preference import model_preference,model_parameter,logger_preference


### TRAIN and TEST ###
def train_loop(model, device, train_loader, criterion,optimizer,epoch) -> None:
    model.train()
    train_loss = 0
    log_interval = 90 # total 185だった
    for batch,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = train_loss / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_loader) ,cur_loss, math.exp(cur_loss)))
            train_loss = 0

def test_loop(model, device, test_loader,criterion,target_class,run):
    model.eval()
    with torch.no_grad():
        target = np.array(())
        output = np.array(())
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat,y)
            target.append(y.cpu().detach().numpy().copy())
            output.append(y_hat.cpu().detach().numpy().copy())

        y_hat_idx = output.max(dim=1).indices
        y_hat_idx = (y_hat_idx == target_class)
        y = (target == target_class)


@click.command()
@click.option('--arch', '-a', help='Name of Architecture')
@click.option('--batch', '-b', default=1000, help='Batch size, default 1000')
@click.option('--minepoch', '-me', default=10, help='Number of min epoches')
@click.option('--learningrate', '-lr', default=1e-2, help='Learning rate')
@click.option('--hidden', '-hidden', default=64, help='dim of hidden layer')
@click.option('--target_class', '-t_class', default=0, help='Target class index')
@click.option('--mode', '-m', default=0, help='0 : normal, 1: best')

def main(arch, batch, minepoch, learningrate,hidden,target_class,mode):
    load_dotenv()
    IDLIST = os.environ['IDLIST']
    FAST5 = os.environ['FAST5']
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
    model,useModel = model_preference(arch,hidden,classes,cutlen,learningrate,target_class,minepoch,heatmap,project_name,mode=mode)
    """
    Training
    """
    ### Train ###
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # network, loss functions and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    print("Train Start...")
    for epoch in range(minepoch):
        train_loop(model, device, train_loader, criterion,optimizer,epoch)

    # testing with validation data
    # print("Test Start...")
    # f1 = test_loop(model, device, test_loader,criterion,target_class)


if __name__ == '__main__':
    main()
