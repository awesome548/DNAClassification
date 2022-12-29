import torch
import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from process import logger_preference,Garbage_collector_callback


@click.command()
@click.option('--target', '-t', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--arch', '-a', help='The path of positive sequence training set')

@click.option('--batch', '-b', default=100, help='Batch size, default 1000')
@click.option('--minepoch', '-me', default=30, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=2e-3, help='Learning rate, default 1e-3')
@click.option('--cutlen', '-len', default=3000, help='Cutting length')
@click.option('--cutoff', '-off', default=1500, help='Cutting length')
@click.option('--classes', '-class', default=6, help='Num of class')
@click.option('--hidden', '-hidden', default=64, help='Num of class')
@click.option('--target_class', '-t_class', default=0, help='Num of class')

def main(target,inpath,arch, batch, minepoch, learningrate,cutlen,cutoff,classes,hidden,target_class):

    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True
    #torch.set_deterministic_debug_mode(True)
    """
    Preference
    """
    project_name = "Category-2-3"
    base_classes = 2
    heatmap = True
    ### Model ###
    model,useModel = model_preference(arch,hidden,classes,cutlen,learningrate,target_class,minepoch,heatmap)
    ### Dataset ###
    dataset_size,cut_size = data_preference(cutoff,cutlen)
    """
    Dataset preparation
    """
    data = Dataformat(target,inpath,dataset_size,cut_size,num_classes=classes,base_classes=base_classes)
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
        #callbacks=[Garbage_collector_callback()],
        #callbacks=[model_checkpoint],
    )
    trainer.fit(model,datamodule=data_module)
    output = trainer.test(model,datamodule=data_module)

    print(type(output))
    #model.state_dict().keys()
    #model.load_from_checkpoint("Baseline-F/2hl41atr/checkpoints/epoch=24-step=2400.ckpt")

if __name__ == '__main__':
    main()
