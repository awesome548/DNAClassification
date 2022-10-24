from pyexpat import model
from unicodedata import bidirectional
from sentry_sdk import configure_scope
import torch
from dataset import Dataset
import click
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import  random_split
import pytorch_lightning as pl
from model import LstmEncoder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datamodule import DataModule

@click.command()
@click.option('--pTrain', '-pt', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--pVal', '-pv', help='The path of positive sequence validation set', type=click.Path(exists=True))
@click.option('--nTrain', '-nt', help='The path of negative sequence training set', type=click.Path(exists=True))
@click.option('--nVal', '-nv', help='The path of negative sequence validation set', type=click.Path(exists=True))
@click.option('--pTest', '-ptt', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--nTest', '-ntt', help='The path of positive sequence validation set', type=click.Path(exists=True))

#@click.option('--outpath', '-o', help='The output path and name for the best trained model')
#@click.option('--interm', '-i', help='The path and name for model checkpoint (optional)', type=click.Path(exists=True), required=False)
@click.option('--batch', '-b', default=200, help='Batch size, default 1000')
@click.option('--epoch', '-e', default=40, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=1e-3, help='Learning rate, default 1e-3')

def main(ptrain, pval, ntrain, nval,ptest,ntest, batch, epoch, learningrate):

    #torch setting
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    ### PREPARATION ###
    #variable
    input_size = 60
    input_length = 50
    hidden_size = 5
    output_size = 2
    lr = learningrate
    size = (input_length,input_size)
    num_classes = 2
    bidirectional = True

    #dataset
    training_set = Dataset(ptrain, ntrain,size)
    validation_set = Dataset(pval, nval,size)
    test_set = Dataset(ptest,ntest,size)
    data_module = DataModule(training_set,validation_set,test_set,batch_size=batch)

    # define logger
    wandb_logger = WandbLogger(project="LSTMtest")

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

    ### TRAINING ###
    model = LstmEncoder(input_size,output_size,hidden_size,lr,num_classes,bidirectional)
    trainer = pl.Trainer(
        max_epochs=epoch,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        callbacks=[model_checkpoint,early_stopping],
    )
    trainer.fit(
        model,
        datamodule=data_module,
    )
    trainer.test(
        model,
        datamodule=data_module,
    )

    
if __name__ == '__main__':
    main()