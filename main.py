import torch
import os
import click
import sys
from dotenv import load_dotenv
import pytorch_lightning as pl
from model import effnetv2,EffNetV2
from ops_data.dataformat import Dataformat
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from preference import model_preference,model_parameter,logger_preference
from pytorch_lightning.loggers import TensorBoardLogger



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
    data_module = data.module(batch)
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
    # refine callbacks
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=10,
    )
    logger = TensorBoardLogger("tb_logs", name="my_model")
    ### Train ###
    trainer = pl.Trainer(
        max_epochs=minepoch,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=logger
        #callbacks=[early_stopping],
        #callbacks=[Garbage_collector_callback()],
        #callbacks=[model_checkpoint],
    )
    trainer.fit(model,datamodule=data_module)
    trainer.test(model,datamodule=data_module)
    #model = EffNetV2.load_from_checkpoint("model_log/Effnet-c2-BC/checkpoints/epoch=19-step=6400.ckpt")


if __name__ == '__main__':
    main()
