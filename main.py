from pyexpat import model
from unicodedata import bidirectional
from xml.etree.ElementInclude import include
from sentry_sdk import configure_scope
import torch
from dataset import FormatDataset,NormalDataset
import click
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import  random_split
import pytorch_lightning as pl
from lstm import LstmEncoder,CNNLstmEncoder
from transformer import ViTransformer
from cnn import ResNet, Bottleneck
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datamodule import DataModule
from preprocess import Preprocess
import glob


@click.command()
@click.option('--pTrain', '-pt', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--pVal', '-pv', help='The path of positive sequence validation set', type=click.Path(exists=True))
@click.option('--nTrain', '-nt', help='The path of negative sequence training set', type=click.Path(exists=True))
@click.option('--nVal', '-nv', help='The path of negative sequence validation set', type=click.Path(exists=True))
@click.option('--pTest', '-ptt', help='The path of positive sequence test set', type=click.Path(exists=True))
@click.option('--nTest', '-ntt', help='The path of negative sequence test set', type=click.Path(exists=True))

#@click.option('--outpath', '-o', help='The output path and name for the best trained model')
#@click.option('--interm', '-i', help='The path and name for model checkpoint (optional)', type=click.Path(exists=True), required=False)
@click.option('--batch', '-b', default=200, help='Batch size, default 1000')
@click.option('--epoch', '-e', default=40, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=1e-3, help='Learning rate, default 1e-3')
@click.option('--cutlen', '-c', default=3000, help='Cutting length')
@click.option('--cutoff', '-co', default=1500, help='Cutting length')


def main(targetA,targetB,targetC,inpath, batch, epoch, learningrate,cutlen,cutoff):

    """
    Dataset Preference
    """
    dataset = glob.glob(inpath+'/*')
    dataset_size = 8000 
    dataA = Preprocess(targetA,dataset[0],cutoff=cutoff).process()
    dataB = Preprocess(targetB,dataset[1],cutoff=cutoff).process()
    dataC = Preprocess(targetC,dataset[2],cutoff=cutoff).process()

    train_size = int(0.8 * dataset_size)
    val_size, test_size = int(0.2 * dataset_size)

    #torch setting
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### PREPARATION ###
    """
    Training setting
    """
    lr = learningrate
    num_classes = 2
    bidirectional = True
    # define logger
    wandb_logger = WandbLogger(project="CutLenSearch")

    """
    MODEL Select
    """
    useResNet = False
    useLstm = True

    """
    Data Format setting
    """
    inputDim = 1
    inputLen = 3000
    hiddenDim = 128
    outputDim = 2
    size = {
        'dim' : inputDim,
        'length' : inputLen,
        'num_class' : num_classes
    }

    """
    CNN setting
    """
    cnn_params = {
        'padd' : 5,
        'ker' : 19,
        'stride' : 3,
        'convDim' : 20,
    }

    """
    LSTM setting
    """
    lstm_params = {
        'inputDim' : 1,
        'hiddenDim' : 128,
        'outputDim' : 2,
        'bidirect' : True
    }


    """
    MODEL architecture
    """
    if useLstm:
            """
            LSTM & CNN
            """
            training_set = FormatDataset(ptrain, ntrain,**size)
            validation_set = FormatDataset(pval, nval,**size)
            test_set = FormatDataset(ptest,ntest,**size)
            data_module = DataModule(training_set,validation_set,test_set,batch_size=batch)
            ### MODEL ###
            model = CNNLstmEncoder(**lstm_params,lr=lr,classes=num_classes,**cnn_params)
    elif useResNet:
        """
        ResNet
        """
        training_set = NormalDataset(ptrain, ntrain,num_classes)
        validation_set = NormalDataset(pval, nval,num_classes)
        test_set = NormalDataset(ptest,ntest,num_classes)
        data_module = DataModule(training_set,validation_set,test_set,batch_size=batch)
        model = ResNet(Bottleneck,[2,2,2,2],classes=num_classes,cutlen=cutlen)


    # refine callbacks
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

    trainer = pl.Trainer(
        max_epochs=epoch,
        min_epochs=20,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        callbacks=[early_stopping],
        # callbacks=[model_checkpoint],
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
