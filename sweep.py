import torch
import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter,logger_preference
from pytorch_lightning.loggers import WandbLogger

def main():
    idpath = "/z/kiku/Dataset/ID"
    inpath ="/z/kiku/Dataset/Target"
    arch = "ResNet"
    batch = 200
    LEARNINGRATE = 2e-3
    cutoff = 1500
    classes = 4
    target = 1
    heatmap = False
    hidden = 64
    EPOCH = 30
    CUTLEN = 3000
    FSTCHAN = 20
    SNDCHAN = 30
    THRCHAN = 45
    FORCHAN = 67
    FSTLAYR = 2
    SNDLAYR = 2
    THRLAYR = 2
    FORLAYR = 2

    cfgs = [
        [FSTCHAN,FSTLAYR],
        [SNDCHAN,SNDLAYR],
        [THRCHAN,THRLAYR],
        [FORCHAN,FORLAYR],
    ]

    """
    Preference
    """
    project_name = "Baseline_resnet_sweep"
    base_classes = 4
    heatmap = True
    ### Model ###
    model,useModel = model_preference(arch,hidden,classes,CUTLEN,LEARNINGRATE,target,EPOCH,heatmap,project_name,cfgs)
    ### Dataset ###
    dataset_size,cut_size = data_preference(cutoff,CUTLEN)
    """
    Dataset preparation
    """
    data = Dataformat(idpath,inpath,dataset_size,cut_size,num_classes=classes,base_classes=base_classes)
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
    wandb_logger = WandbLogger(
        project=project_name,
        config={
            "dataset_size" : dataset_size,
            "model" : useModel,
            "cutlen" : CUTLEN,
            "epoch" : EPOCH,
            "learningrate" : LEARNINGRATE,
            "conv_1" : FSTCHAN,
            "conv_2" : SNDCHAN,
            "conv_3" : THRCHAN,
            "conv_4" : FORCHAN,
            "layer_1" : FSTLAYR,
            "layer_2" : SNDLAYR,
            "layer_3" : THRLAYR,
            "layer_4" : FORLAYR,
        },
        name=useModel+"_"+str(classes)+"_"+str(CUTLEN)+"_e_"+str(EPOCH)+"_"+str(target)
    )
    ### Train ###
    trainer = pl.Trainer(
        max_epochs=EPOCH,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        #callbacks=[Garbage_collector_callback()],
        callbacks=[early_stopping],
    )
    trainer.fit(model,datamodule=data_module)
    trainer.test(model,datamodule=data_module)

if __name__ == '__main__':
    main()
