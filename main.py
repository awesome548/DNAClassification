import torch
import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import effnetv2,EffNetV2
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter,logger_preference


@click.command()
@click.option('--idpath', '-id', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--arch', '-a', help='The path of positive sequence training set')

@click.option('--batch', '-b', default=100, help='Batch size, default 1000')
@click.option('--minepoch', '-me', default=40, help='Number of epoches, default 20')
@click.option('--learningrate', '-lr', default=2e-3, help='Learning rate, default 1e-3')
@click.option('--cutlen', '-len', default=3000, help='Cutting length')
@click.option('--cutoff', '-off', default=1500, help='Cutting length')
@click.option('--classes', '-class', default=7, help='Num of class')
@click.option('--hidden', '-hidden', default=64, help='Num of class')
@click.option('--target_class', '-t_class', default=0, help='Num of class')
@click.option('--mode', '-m', default=0, help='0 : normal, 1: best')

def main(idpath,inpath,arch, batch, minepoch, learningrate,cutlen,cutoff,classes,hidden,target_class,mode):

    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True
    #torch.set_deterministic_debug_mode(True)
    """
    Preference
    """
    project_name = "2Stage-Analysis"
    heatmap = False
    cfgs =[
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  6, 1, 1],
        [6, 256,  6, 2, 1],
    ]
    ### Model ###
    model,useModel = model_preference(arch,hidden,classes,cutlen,learningrate,target_class,minepoch,heatmap,project_name,mode=mode)
    ### Dataset ###
    dataset_size,cut_size = data_preference(cutoff,cutlen)
    """
    Dataset preparation
    """
    data = Dataformat(idpath,inpath,dataset_size,cut_size,num_classes=classes)
    data_module = data.module(batch)
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
    ### Logger ###
    wandb_logger = logger_preference(project_name,classes,dataset_size,useModel,cutlen,minepoch,target_class) 
    ### Train ###
    trainer = pl.Trainer(
        max_epochs=minepoch,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        #callbacks=[early_stopping],
        #callbacks=[Garbage_collector_callback()],
        #callbacks=[model_checkpoint],
    )
    trainer.fit(model,datamodule=data_module)
    #model = EffNetV2.load_from_checkpoint("model_log/Effnet-c2-BC/checkpoints/epoch=19-step=6400.ckpt")
    trainer.test(model,datamodule=data_module)


if __name__ == '__main__':
    main()
