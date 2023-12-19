import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ML_dataset import Dataformat
from ML_model.preference import model_preference,data_preference,model_parameter,logger_preference
from pytorch_lightning.loggers import WandbLogger
import wandb

hyperparameter_defaults = dict(
    cutlen=3000,
    learningrate=2e-3,
    conv_1=20,
    conv_2=30,
    conv_3=45,
    conv_4=67,
    layer_1=2,
    layer_2=2,
    layer_3=2,
    layer_4=2,
)

wandb.init(config=hyperparameter_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config

def main(config):
    EPOCH = 30
    idpath = "/z/kiku/Dataset/ID"
    inpath ="/z/kiku/Dataset/Target"
    arch = "ResNet"
    batch = 256
    cutoff = 1500
    classes = 4
    target = 1
    heatmap = False
    hidden = 64
    project_name = "Baseline_resnet_sweep"
    base_classes = 4
    heatmap = True

    LEARNINGRATE = config.learningrate
    CUTLEN = config.cutlen
    conv_1 = config.conv_1
    conv_2 = config.conv_2
    conv_3 = config.conv_3
    conv_4 = config.conv_4
    layer_1 = config.layer_1
    layer_2 = config.layer_2
    layer_3 = config.layer_3
    layer_4 = config.layer_4

    cfgs = [
        [conv_1,layer_1],
        [conv_2,layer_2],
        [conv_3,layer_3],
        [conv_4,layer_4],
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
