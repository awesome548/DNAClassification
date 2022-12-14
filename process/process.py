import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MyProcess(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y)

        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y)

        self.log("valid_loss",loss)
        return {"valid_loss" : loss}

    def test_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x,text="test")

        ### Loss ###
        y_float =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y_float)
        self.log("test_loss",loss)
        

        ### Kmeans ###
        self.labels = torch.vstack((self.labels,y.detach()))

        ### Metrics ###
        y_hat_idx = y_hat.max(dim=1).indices
        acc = (y == y_hat_idx).float().mean().item()
        y_hat_idx = (y_hat_idx == 0)
        y = (y == 0)
        self.metric['tp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
        self.metric['fp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx != y))
        self.metric['tn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx == y))
        self.metric['fn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))

        self.acc = np.append(self.acc,acc)

        ### Kmeans ###
        #self.position = np.append(self.posiiton,y_hat)
        #self.labels = np.append(self.labels,y)
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

        ### K-Means ###
        cluster = self.cluster[1:,]
        X = cluster.cpu().detach().numpy().copy()
        labels = self.labels[1:,]
        #X = np.reshape(X,(-1,self.hiddenDim*2))

        #position = torch.from_numpy((self.position).astype(np.float32)).clone()
        #labels = torch.from_numpy((self.labels).astype(np.float32)).clone()

        #position = torch.cat((torch.reshape(labels,(-1,1)),position),dim=1)
        kmeans = KMeans(n_clusters=self.classes,random_state=0).fit(X)

        linear = nn.Linear(self.hiddenDim*2,2)
        position = linear(cluster)
        position = torch.cat((labels,position),dim=1)
        assert position.shape[1] == 3

        marker = [".","*","+","x","o","^"]
        color = ['r','b','g','c','m','y']
        """
        position = [labels,pos1,pos2]
        kmeans.labels_ = [k-labels]
        """
        for i in range(self.classes):
            p = position[position[0]==i,:]
            plt.scatter(position[:,1],position[:,2],marker=marker[i],color=color[i])
        plt.savefig("kmeans.png")
        plt.clf()
        for i in range(self.classes):
            p = position[kmeans.labels_ ==i,:]
            plt.scatter(p[:,1],p[:,2],marker = marker[i],color = color[i])
        plt.savefig("kmeans-2.png")