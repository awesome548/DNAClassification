from pyexpat import model
from sentry_sdk import configure_scope
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import math
import numpy as np
from dataset import Dataset
import click
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from model import LstmEncoder
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class DataModule(pl.LightningDataModule):
    def __init__(self,train,val, batch_size: int):
        super().__init__()
        self.train_datasets = train
        self.val_datasets = val
        # self.test_datasets = test   
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(self.train_datasets,batch_size=self.batch_size,shuffle=True,num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.val_datasets,batch_size=self.batch_size,num_workers=24)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset,batch_size=self.batch_size)
        
@click.command()
@click.option('--pTrain', '-pt', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--pVal', '-pv', help='The path of positive sequence validation set', type=click.Path(exists=True))
@click.option('--nTrain', '-nt', help='The path of negative sequence training set', type=click.Path(exists=True))
@click.option('--nVal', '-nv', help='The path of negative sequence validation set', type=click.Path(exists=True))
#@click.option('--outpath', '-o', help='The output path and name for the best trained model')
#@click.option('--interm', '-i', help='The path and name for model checkpoint (optional)', type=click.Path(exists=True), required=False)
@click.option('--batch', '-b', default=200, help='Batch size, default 1000')
@click.option('--epoch', '-e', default=40, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=1e-3, help='Learning rate, default 1e-3')

def main(ptrain, pval, ntrain, nval, batch, epoch, learningrate):

    #torch setting
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    #variable
    input_size = 3000
    hidden_size = 5
    output_size = 2
    lr = learningrate

    #dataset
    training_set = Dataset(ptrain, ntrain)
    validation_set = Dataset(pval, nval)
    data_module = DataModule(training_set,validation_set, batch_size=batch)

    #train
    # define logger
    tensorbord_logger = pl_loggers.TensorBoardLogger("logs/")

    # define callbacks

    model_checkpoint = ModelCheckpoint(
        "logs/",
        filename="{epoch}-{valid_loss:.4f}",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=10,
    )

    model = LstmEncoder(input_size,hidden_size,output_size,lr)
    trainer = pl.Trainer(
        max_epochs=epoch,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=[tensorbord_logger],
        callbacks=[model_checkpoint,early_stopping],
    )
    trainer.fit(
        model,
        datamodule=data_module,
    )

    
if __name__ == '__main__':
    main()