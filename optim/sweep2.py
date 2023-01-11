import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter,logger_preference
from pytorch_lightning.loggers import WandbLogger
import wandb

hyperparameter_defaults = dict(
    cutlen=3000,
    learningrate=2e-3,
    channel=24,
    kernel=19,
    stride=1,
    padd=0,
    mode=0,
)

EPOCH = 35
IDPATH = "/z/kiku/Dataset/ID"
INPATH ="/z/kiku/Dataset/Target"
ARCH = "Effnet"
BATCH = 100
CUTOFF = 1500
CLASSES = 4
TARGET = 1
HIDDEN = None
PROJECT = "Baseline4-effnet-sweep"
HEATMAP = False

wandb.init(project=PROJECT,config=hyperparameter_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


def main(config):
    lr = config.learningrate
    cutlen = config.cutlen
    channel = config.channel
    kernel = config.kernel
    stride = config.stride
    padd = config.padd
    mode = config.mode

    params = {
        "channel" : channel,
        "kernel" : kernel,
        "stride" : stride,
        "padd" : padd,
    }

    ### Model ###
    model,useModel = model_preference(ARCH,HIDDEN,CLASSES,cutlen,lr,TARGET,EPOCH,HEATMAP,PROJECT,mode=mode,cnn_params=params)
    ### Dataset ###
    dataset_size,cut_size = data_preference(CUTOFF,cutlen)
    """
    Dataset preparation
    """
    data = Dataformat(IDPATH,INPATH,dataset_size,cut_size,num_classes=CLASSES)
    data_module = data.module(BATCH)
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
        project=PROJECT,
        config={
            "dataset_size" : dataset_size,
            "model" : useModel,
            "epoch" : EPOCH,
        },
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
    print(f'Starting a run with {config}')
    main(config)
