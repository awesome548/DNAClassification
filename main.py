import torch
import click
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from models import CNNLstmEncoder,ResNet,Bottleneck,SimpleViT,ViT,ViT2,SimpleViT2
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import glob
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from process import logger_preference


@click.command()
@click.option('--target', '-t', help='The path of positive sequence training set', type=click.Path(exists=True))
# Inpath is the taget directory of all dataset
@click.option('--inpath', '-i', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--arch', '-a', help='The path of positive sequence training set')

@click.option('--batch', '-b', default=100, help='Batch size, default 1000')
@click.option('--minepoch', '-me', default=30, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=1e-2, help='Learning rate, default 1e-3')
@click.option('--cutlen', '-len', default=3000, help='Cutting length')
@click.option('--cutoff', '-off', default=1500, help='Cutting length')
@click.option('--classes', '-class', default=3, help='Num of class')
@click.option('--hidden', '-hidden', default=64, help='Num of class')


def main(target,inpath,arch, batch, minepoch, learningrate,cutlen,cutoff,classes,hidden):

    """
    Preference
    """
    project_name = "Baseline-F"
    ### Model ###
    cnn_params,lstm_params,transformer_params = model_parameter(classes,hidden)
    model,transform,useModel = model_preference(arch,lstm_params,transformer_params,classes,cutlen,learningrate)
    ### Dataset ###
    base_classes,dataset_size,data_transform,cut_size = data_preference(transform,cutoff,cutlen)
    """
    Dataset
    """
    idset = glob.glob(target+'/*.txt')
    dataset = glob.glob(inpath+'/*')

    data = Dataformat(idset,dataset,dataset_size,cut_size,num_classes=classes,base_classes=base_classes,transform=data_transform)
    data_module = data.process(batch)
    dataset_size = data.size()

    """
    Training
    """
    # refine callbacks
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=10,
    )
    ### Torch setting ###
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ### Logger ###
    wandb_logger = logger_preference(project_name,classes,dataset_size,useModel,cutlen,minepoch) 
    epoch = minepoch + 5
    ### Train ###
    trainer = pl.Trainer(
        max_epochs=epoch,
        min_epochs=minepoch,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        #callbacks=[early_stopping],
        #callbacks=[model_checkpoint],
    )
    #trainer.fit(model,datamodule=data_module)
    model.state_dict().keys()
    model.load_from_checkpoint("Baseline-F/2hl41atr/checkpoints/epoch=24-step=2400.ckpt")
    trainer.test(model,datamodule=data_module)


if __name__ == '__main__':
    main()
