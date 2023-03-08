import torch
import numpy as np
from dataset.in_category_data import Dataformat2
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from model import LSTM,resnet,SimpleViT,ViT,ViT2,SimpleViT2,Transformer_clf_model,myGRU,EffNetV2,ResNet
import torch.utils.data
import wandb
from torchmetrics import Accuracy,Recall,Precision,F1Score,ConfusionMatrix,AUROC


def category_loop(model, device, test_loader,n_cls,tar_cls,run):
    y_hat_idx = torch.zeros(1).to(device)
    y = torch.zeros(1).to(device)
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)

            pred_idx = pred.max(dim=1).indices
            y_hat_idx = torch.cat((y_hat_idx,pred_idx))
            y = torch.cat((y,target))

        y_hat_idx = y_hat_idx[1:]
        y = y[1:]
        acc = Accuracy(task="multiclass",num_classes=n_cls).to(device)
        acc1 = Accuracy(task="multiclass",num_classes=n_cls,average=None).to(device)
        preci = Precision(task="multiclass",num_classes=n_cls).to(device)
        preci1 = Precision(task="multiclass",num_classes=n_cls,average=None).to(device)
        recall = Recall(task="multiclass",num_classes=n_cls).to(device)
        recall1 = Recall(task="multiclass",num_classes=n_cls,average=None).to(device)
        run.log({
            'Metric_AccuracyMacro' : acc(y_hat_idx,y),
            'Metric_AccuracyMicro' : acc1(y_hat_idx,y)[tar_cls],
            'Metric_RecallMacro' : recall(y_hat_idx,y),
            'Metric_RecallMicro' : recall1(y_hat_idx,y)[tar_cls],
            'Metric_PrecisionMacro' : preci(y_hat_idx,y),
            'Metric_PrecisionMicro' : preci1(y_hat_idx,y)[tar_cls],
        })

    return y_hat_idx == tar_cls

def in_category_loop(model, device, test_loader,n_cls,tar_cls,run):
    y_hat_idx = torch.zeros(1).to(device)
    y = torch.zeros(1).to(device)
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)

            pred_idx = pred.max(dim=1).indices
            y_hat_idx = torch.cat((y_hat_idx,pred_idx))
            y = torch.cat((y,target))

        y_hat_idx = y_hat_idx[1:]
        y = y[1:]
        assert y.shape == y_hat_idx.shape

        preci1 = Precision(task="multiclass",num_classes=n_cls,average=None).to(device)
        recall1 = Recall(task="multiclass",num_classes=n_cls,average=None).to(device)
        confmat = ConfusionMatrix(task="multiclass",num_classes=n_cls).to(device)
        run.log({
            'Metric_RecallMicro' : recall1(y_hat_idx,y)[tar_cls],
            'Metric_PrecisionMicro' : preci1(y_hat_idx,y)[tar_cls],
        })
        confmat = confmat(y_hat_idx, y)
        print(confmat)

### varible ###
IDPATH = "/z/kiku/Dataset/ID"
INPATH ="/z/kiku/Dataset/Target"
ARCH = "Effnet"
BATCH = 100
EPOCH = 20
CUTOFF = 1500
HEATMAP = False
LR = 0.02
PROJECT = "2Stage-Analysis"

if __name__ == "__main__":
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    CATEGORY
    """
    run = wandb.init(
        project=PROJECT,
        reinit=True,
    )
    cls = 5
    tar_cls = 1
    cut = 3000
    with run:
        #model,useModel = model_preference("GRU",hidden=128,classes=cls,cutlen=cut,learningrate=0.001,target=tar_cls,epoch=40,heatmap=HEATMAP,project=PROJECT)
        model = myGRU.load_from_checkpoint("2Stage-Analysis/3o5kuesn/checkpoints/epoch=39-step=14960.ckpt")
        model = model.to(device)
        ### Dataset ###
        dataset_size,cut_size = data_preference(CUTOFF,cut)
        data = Dataformat(IDPATH,INPATH,dataset_size,cut_size,num_classes=5)
        test_loader = data.test_loader(BATCH)

        # testing with validation data
        print("Test Start...")
        tar_idx = category_loop(model, device, test_loader,cls,tar_cls,run)

    """
    IN CATEGORY
    """
    run = wandb.init(
        project=PROJECT,
        reinit=True,
    )
    cls = 2
    tar_cls = 0
    cut = 9000
    with run:
        model = EffNetV2.load_from_checkpoint("model_log/Effnet-c2-BC/checkpoints/epoch=19-step=6400.ckpt")
        model = model.to(device)
        ### Dataset ###
        dataset_size,cut_size = data_preference(CUTOFF,cut)
        data = Dataformat2(IDPATH,INPATH,dataset_size,cut_size,num_classes=5,idx=tar_idx)
        test_loader = data.test_loader(BATCH)

        # testing with validation data
        print("Test Start...")
        in_category_loop(model, device, test_loader,cls,tar_cls,run)