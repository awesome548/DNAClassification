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
    cfgs="big"
)

wandb.init(config=hyperparameter_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config

def main(config):
    EPOCH = 30
    idpath = "/z/kiku/Dataset/ID"
    inpath ="/z/kiku/Dataset/Target"
    arch = "Effnet"
    batch = 64
    cutoff = 1500
    classes = 2
    target = 1
    hidden = None
    project_name = "Category-23-optim"
    base_classes = 2
    heatmap = False

    LEARNINGRATE = config.learningrate
    CUTLEN = config.cutlen
    CHAN = config.channel
    KERNEL = config.kernel
    STRIDE = config.stride
    PADD = config.padd
    cfgs = config.cfgs

    if cfgs =="big":
        cfgs = [
            # t, c, n, s, SE
            [1,  24,  2, 1, 0],
            [4,  48,  4, 2, 0],
            [4,  64,  4, 2, 0],
            [4, 128,  6, 2, 1],
            [6, 160,  6, 1, 1],
            [6, 256,  6, 2, 1],
            [CHAN,KERNEL,STRIDE,PADD],
        ]
    else:
        cfgs = [
            # t, c, n, s, SE
            [1,  24,  2, 1, 0],
            [4,  48,  4, 2, 0],
            [4,  64,  4, 2, 0],
            [4, 128,  4, 2, 1],
            [6, 160,  4, 1, 1],
            [6, 256,  4, 2, 1],
            [CHAN,KERNEL,STRIDE,PADD],
        ]


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
