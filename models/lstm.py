import torch.nn as nn
import torch
from process import MyProcess 
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class LSTM(pl.LightningModule):
    def __init__(self,hiddenDim,lr,classes,bidirect,target):
        super(LSTM,self).__init__()
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.classes = classes
        self.target = target
        """
        ResNet conv
        """
        self.conv = nn.Sequential(
            nn.Conv1d(1,20,19, padding=5, stride=3),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=1, stride=2),
        )
        #Model Architecture
        if bidirect:
            self.lstm = nn.LSTM(input_size = 20,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            self.fc = nn.Linear(hiddenDim*2, classes)
            self.hiddenDim = hiddenDim*2
        else:
            self.lstm = nn.LSTM(input_size = 20,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            )
            self.fc = nn.Linear(hiddenDim, classes)
            self.hiddenDim = hiddenDim

        self.acc = np.array([])
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        self.labels = torch.zeros(1).cuda()
        self.cluster = torch.zeros(1,hiddenDim*2).cuda()
        self.save_hyperparameters()

    def forward(self,x,hidden0=None,text=None):
        x = self.conv(x.unsqueeze(1))
        output, (hidden,cell) = self.lstm(torch.transpose(x,1,2),hidden0)
        y_hat = self.fc(output[:,-1,])
        #if text == "test":
            #with torch.no_grad():
                #self.cluster = torch.vstack((self.cluster,output[:,-1,].detach().clone().cpu()))
        return y_hat

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
        x, y = batch
        y_hat = self.forward(x,text="test")

        ### Loss ###
        y_float =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y_float)
        self.log("test_loss",loss)
        

        ### Kmeans ###
        self.labels = torch.hstack((self.labels,y.clone().detach()))

        ### Metrics ###
        target = self.target
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
        n_class = self.classes
        target = str(self.target)

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
        #cluster = self.cluster[1:,].cuda()
        #labels = self.labels[1:]
        #X = cluster.cpu().detach().numpy().copy()

        #kmeans = KMeans(n_clusters=n_class,init='k-means++',n_init=1,random_state=0).fit(X)
        
        #heat_map = torch.zeros(n_class,n_class)
        #val_len = 0
        #for i in range(n_class):
            #p = labels[kmeans.labels_ ==i]
            #val_len += int(p.shape[0])
            #for j in range(n_class):
                #x = torch.zeros(p.shape)
                #x[p==j] = 1
                #heat_map[i,j] = torch.count_nonzero(x)
        #assert val_len == int(labels.shape[0])
        #heatmap = (heat_map/20).cpu().detach().numpy().copy()
        ##heatmap = (heat_map/20)
        #plt.figure()
        #s = sns.heatmap(heatmap,annot=True,cmap="Reds",fmt=".3g")
        #s.set(xlabel="label",ylabel="cluster")
        #plt.savefig(f"heatmap-transformer-{target}.png")
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer