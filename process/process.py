import pytorch_lightning as pl
import torch
import os
import numpy as np
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Accuracy,Recall,Precision,F1Score,ConfusionMatrix,AUROC

class MyProcess(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)

        self.log("train_loss",loss)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)

        self.log("valid_loss",loss)
        return {"valid_loss" : loss}
    
    def on_test_start(self) -> None:
        self.start_time = time.perf_counter()
        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x,text="test")

        ### Loss ###
        loss = self.loss_fn(y_hat,y)
        self.log("test_loss",loss)

        ### Kmeans ###
        if self.pref["heatmap"]:
            self.labels = torch.hstack((self.labels,y.clone().detach()))
        ### Metrics ###
        y_hat_idx = y_hat.max(dim=1).indices

        acc = (y == y_hat_idx).float().mean().item()
        self.acc = np.append(self.acc,acc)

        return {'test_loss':loss,'preds':y_hat,'preds_idx':y_hat_idx,'target':y}

    def test_epoch_end(self,outputs):
        self.end_time = time.perf_counter()
        inference_time = self.end_time - self.start_time
        self.log("inference_time",inference_time)

        ### Valuables ###
        _,cutlen,n_class,epoch,target,name,heatmap,project = self.pref.values()
        ### Merics ###
        y_hat = outputs[0]['preds']
        y_hat_idx = outputs[0]['preds_idx']
        y = outputs[0]['target']

        for i in range(len(outputs)-1):
            i +=1
            y_hat = torch.vstack((y_hat,outputs[i]['preds']))
            y_hat_idx = torch.hstack((y_hat_idx,outputs[i]['preds_idx']))
            y = torch.hstack((y,outputs[i]['target']))

        y_hat_idx = y_hat.cpu()
        y_hat = y_hat.cpu()
        y = y.cpu()
        acc = Accuracy(task="multiclass",num_classes=n_class)
        acc1 = Accuracy(task="multiclass",num_classes=n_class,average=None)
        preci = Precision(task="multiclass",num_classes=n_class)
        preci1 = Precision(task="multiclass",num_classes=n_class,average=None)
        recall = Recall(task="multiclass",num_classes=n_class)
        recall1 = Recall(task="multiclass",num_classes=n_class,average=None)
        f1 = F1Score(task="multiclass",num_classes=n_class,average=None)
        auroc = AUROC(task="multiclass", num_classes=n_class)
        auroc1 = AUROC(task="multiclass", num_classes=n_class,average=None)
        confmat = ConfusionMatrix(task="multiclass",num_classes=n_class)
        self.log_dict({
            "Self_AccuracyMacro" : self.acc.mean(),
            'Metric_AccuracyMacro' : acc(y_hat_idx,y),
            'Metric_AccuracyMicro' : acc1(y_hat_idx,y)[target],
            'Metric_RecallMacro' : recall(y_hat_idx,y),
            'Metric_RecallMicro' : recall1(y_hat_idx,y)[target],
            'Metric_PrecisionMacro' : preci(y_hat_idx,y),
            'Metric_PrecisionMicro' : preci1(y_hat_idx,y)[target],
            'Metric_F1' : f1(y_hat_idx,y)[target],
            'Metric_AurocMacro' : auroc(y_hat,y),
            'Metric_AurocMicro' : auroc1(y_hat,y)[target],
        })
        confmat = confmat(y_hat_idx, y)


        ### K-Means ###
        if heatmap:
            cluster = self.cluster[1:,]
            labels = self.labels[1:]
            X = cluster.cpu().detach().numpy().copy()
            heat_map = torch.zeros(n_class,n_class)

            kmeans = KMeans(n_clusters=n_class,init='k-means++',n_init=1,random_state=0).fit(X)

            val_len = 0
            for i in range(n_class):
                p = labels[kmeans.labels_ ==i]
                val_len += int(p.shape[0])
                for j in range(n_class):
                    x = torch.zeros(p.shape)
                    x[p==j] = 1
                    heat_map[i,j] = torch.count_nonzero(x)

            assert val_len == int(labels.shape[0])
            for i in range(n_class):
                heat_map[:,i] = heat_map[:,i]/heat_map.sum(0)[i]
            heatmap = heat_map.cpu().detach().numpy().copy()

            os.makedirs(f"heatmaps/{project}",exist_ok=True)
            ### SAVE FIG ###
            plt.figure()
            s = sns.heatmap(heatmap,vmin=0.0,vmax=1.0,annot=True,cmap="Reds",fmt=".3g")
            s.set(xlabel="label",ylabel="cluster")
            plt.savefig(f"heatmaps/{project}/{name}-{str(cutlen)}-e{epoch}-c{n_class}-{inference_time}.png")
            ### SAVE FIG ###
        confmat = confmat.cpu().detach().numpy().copy()
        os.makedirs(f"confmat/{project}",exist_ok=True)
        plt.figure()
        s = sns.heatmap(confmat,annot=True,cmap="Reds",fmt="d")
        s.set(xlabel="predicted",ylabel="label")
        plt.savefig(f"confmat/{project}/{name}-{str(cutlen)}-e{epoch}-c{n_class}-{inference_time}.png")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer
