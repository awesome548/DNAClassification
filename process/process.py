import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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
        y_float =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y_float)
        self.log("test_loss",loss)
        

        ### Kmeans ###
        self.labels = torch.hstack((self.labels,y.clone().detach().cpu()))

        ### Metrics ###
        target = 0
        y_hat_idx = y_hat.max(dim=1).indices
        acc = (y == y_hat_idx).float().mean().item()
        y_hat_idx = (y_hat_idx == target)
        y = (y == target)
        self.metric['tp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
        self.metric['fp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx != y))
        self.metric['tn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx == y))
        self.metric['fn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))

        self.acc = np.append(self.acc,acc)

        return {"test_loss" : loss}

    def test_epoch_end(self,outputs):

        ### Merics ###
        tp,fp,fn,tn = self.metric.values()
        acc = self.acc.mean()
        acc_1 = (tp+tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)

        self.log("test_Accuracy",acc)
        self.log("test_Accuracy2",acc_1)
        self.log("test_Recall",recall)
        self.log("test_Precision",precision)

        """
        ### K-Means ###
        ### tensor
        cluster = self.cluster[1:,].cuda()
        X = cluster.cpu().detach().numpy().copy()
        labels = self.labels[1:]

        assert cluster.shape[1] == 128
        assert labels.shape[0] == 2000*6
        ### numpy
        #X = self.cluster.reshape((-1,self.hiddenDim))
        #labels = self.labels

        kmeans = KMeans(n_clusters=6,init='k-means++',n_init=1,random_state=0).fit(X)

        #linear = nn.Linear(self.hiddenDim*2,2).cuda()
        #position = linear(cluster)

        marker = [".","*","+","x","o","^"]
        color = ['r','b','g','c','m','y']
        
        heat_map = torch.zeros(6,6)
        #heat_map = np.zeros((6,6))
        val_len = 0
        for i in range(6):
            p = labels[kmeans.labels_ ==i]
            val_len += int(p.shape[0])
            for j in range(6):
                #print(p.shape)
                x = torch.zeros(p.shape)
                #x = np.zeros(p.shape)
                x[p==j] = 1
                heat_map[i,j] = torch.count_nonzero(x)
                #heat_map[i,j] = np.count_nonzero(x)
        assert val_len == int(labels.shape[0])
        heatmap = (heat_map/20).cpu().detach().numpy().copy()
        #heatmap = (heat_map/20)
        plt.figure()
        s = sns.heatmap(heatmap,annot=True,cmap="Reds",fmt=".3g")
        s.set(xlabel="label",ylabel="cluster")
        plt.savefig("heatmap.png")
        
        #for i in range(self.classes):
            #p = position[kmeans.labels_ ==i,:]
            #plt.scatter(p[:,1],p[:,2],marker = marker[i],color = color[i])
            #plt.savefig("kmeans-2.png")
        
        """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer