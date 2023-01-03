import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import ParserError

class MyProcess(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)

        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)

        self.log("valid_loss",loss)
        return {"valid_loss" : loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x,text="test")

        ### Loss ###
        loss = self.loss_fn(y_hat,y)
        self.log("test_loss",loss)
        
        ### Kmeans ###
        self.labels = torch.hstack((self.labels,y.clone().detach()))

        ### Metrics ###
        target = self.pref["target"]
        y_hat_idx = y_hat.max(dim=1).indices
        acc = (y == y_hat_idx).float().mean().item()
        self.acc = np.append(self.acc,acc)

        y_hat_idx = (y_hat_idx == target)
        y = (y == target)
        self.metric['tp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
        self.metric['fp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx != y))
        self.metric['tn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx == y))
        self.metric['fn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))

        return {"test_loss" : loss}

    def test_epoch_end(self,outputs):
        ### Valuables ###
        _,cutlen,n_class,epoch,_,name,heatmap,project = self.pref.values()
        ### Merics ###
        tp,fp,fn,tn = self.metric.values()
        self.log("test_Accuracy",self.acc.mean())
        self.log("test_Accuracy2",(tp+tn)/(tp+tn+fp+fn))
        self.log("test_Recall",(tp)/(tp+fn))
        self.log("test_Precision",(tp)/(tp+fp))

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

            ### SAVE FIG ###
            plt.figure()
            s = sns.heatmap(heatmap,annot=True,cmap="Reds",fmt=".3g")
            s.set(xlabel="label",ylabel="cluster")
            plt.savefig(f"heatmaps/{project}/{name}-{str(cutlen)}-e{epoch}.png")

        return outputs
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer