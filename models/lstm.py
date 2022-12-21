import torch.nn as nn
import torch
from process import MyProcess 
import numpy as np

class LSTM(MyProcess):
    def __init__(self,hiddenDim,lr,classes,bidirect,target,cutlen,epoch):
        super(LSTM,self).__init__()
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.classes = classes
        self.target = target
        self.cutlen = cutlen
        self.epoch = epoch

        dim = 20
        self.conv = nn.Sequential(
            nn.Conv1d(1,dim,19, padding=5, stride=3),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=1, stride=2),
        )
        #Model Architecture
        if bidirect:
            self.lstm = nn.LSTM(input_size = dim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            self.fc = nn.Linear(hiddenDim*2, classes)
            self.hiddenDim = hiddenDim*2
        else:
            self.rnn = nn.LSTM(input_size = dim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = False,
                            )
            self.fc = nn.Linear(hiddenDim, classes)

        self.acc = np.array([])
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        self.labels = torch.zeros(1).cuda()
        self.cluster = torch.zeros(1,hiddenDim).cuda()
        self.save_hyperparameters()

    def forward(self,x,hidden0=None,text=None):
        x = x.unsqueeze(1)
        x = self.conv(x)
        output, (hidden,cell) = self.lstm(torch.transpose(x,1,2),hidden0)
        y_hat = self.fc(output[:,-1,])
        if text == "test":
            self.cluster = torch.vstack((self.cluster,output[:,-1,].detach().clone()))
        return y_hat
