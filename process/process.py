import pytorch_lightning as pl
import torch
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score
from torchmetrics import Accuracy,Recall,Precision,F1Score,ConfusionMatrix

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x,text="test")
        y_hat_idx = y_hat.max(dim=1).indices

        ### Loss ###
        loss = self.loss_fn(y_hat,y)
        self.log("test_loss",loss)

        ### Kmeans ###
        if self.pref["heatmap"]:
            self.labels = torch.hstack((self.labels,y.clone().detach()))
        ### Metrics ###
        """
        target = self.pref["target"]
        acc = (y == y_hat_idx).float().mean().item()
        self.acc = np.append(self.acc,acc)

        y_hat_idx = (y_hat_idx == target)
        y = (y == target)
        self.metric['tp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
        self.metric['fp'] += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx != y))
        self.metric['tn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx == y))
        self.metric['fn'] += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))
        """
        return {'test_loss':loss,'preds':y_hat_idx,'target':y}

    def test_epoch_end(self,outputs):
        ### Valuables ###
        _,cutlen,n_class,epoch,target,name,heatmap,project = self.pref.values()
        ### Merics ###
        """
        tp = self.metric['tp']
        fp = self.metric['fp']
        tn = self.metric['tn']
        fn = self.metric['fn']
        self.log("test_Accuracy",self.acc.mean())
        self.log("test_Accuracy2",(tp+tn)/(tp+tn+fp+fn))
        self.log("test_Recall",(tp)/(tp+fn))
        self.log("test_Precision",(tp)/(tp+fp))
        self.log("test_F1",2*( (tp)/(tp+fp) * (tp)/(tp+fn) ) / ( (tp)/(tp+fp) + (tp)/(tp+fn) ))
        """
        print(outputs)
        y_hat = outputs[0]['preds']
        y = outputs[0]['target']

        for i in range(len(outputs)-1):
            i +=1
            y_hat = torch.hstack((y_hat,outputs[i]['preds']))
            y = torch.hstack((y,outputs[i]['target']))

        y_hat = y_hat.cpu()
        y = y.cpu()
        acc = Accuracy(task="multiclass",num_classes=n_class)
        acc1 = Accuracy(task="multiclass",num_classes=n_class,average=None)
        preci = Precision(task="multiclass",num_classes=n_class,average=None)
        recall = Recall(task="multiclass",num_classes=n_class,average=None)
        f1 = F1Score(task="multiclass",num_classes=n_class,average=None)
        confmat = ConfusionMatrix(task="multiclass",num_classes=n_class)
        self.log_dict({
            'test_macro_Accuracy' : acc(y_hat,y),
            'test_micro_Accuracy' : acc1(y_hat,y)[target],
            'test_Recall' : recall(y_hat,y)[target],
            'test_Precision' : preci(y_hat,y)[target],
            'test_F1' : f1(y_hat,y)[target],
        })
        confmat = confmat(y_hat, y)


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
            confmat = confmat.cpu().detach().numpy().copy()

            os.makedirs(f"heatmaps/{project}",exist_ok=True)
            os.makedirs(f"confmat/{project}",exist_ok=True)
            ### SAVE FIG ###
            plt.figure()
            s = sns.heatmap(heatmap,vmin=0.0,vmax=1.0,annot=True,cmap="Reds",fmt=".3g")
            s.set(xlabel="label",ylabel="cluster")
            plt.savefig(f"heatmaps/{project}/{name}-{str(cutlen)}-e{epoch}.png")
            ### SAVE FIG ###
            plt.figure()
            s = sns.heatmap(confmat,annot=True,cmap="Reds",fmt="d")
            s.set(xlabel="label",ylabel="cluster")
            plt.savefig(f"confmat/{project}/{name}-{str(cutlen)}-e{epoch}.png")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer
