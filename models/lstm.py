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

        self.acc = np.array([])
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        self.save_hyperparameters()

    def forward(self,x,hidden0=None,text=None):
        x = self.conv(x.unsqueeze(1))
        output, (hidden,cell) = self.lstm(torch.transpose(x,1,2),hidden0)
        y_hat = self.fc(output[:,-1,])
        #if text == "test":
            #with torch.no_grad():
                #self.cluster = torch.vstack((self.cluster,output[:,-1,].detach().clone().cpu()))
        return y_hat
