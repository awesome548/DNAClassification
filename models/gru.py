from process import MyProcess 
import torch.nn as nn
import torch
import numpy as np

class GRU(MyProcess):
    def __init__(self,cnn_params,hiddenDim,lr,classes,bidirect,target,cutlen,epoch,name,heatmap):
        super(GRU,self).__init__()
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.pref = {
            "classes" : classes,
            "target" : target,
            "cutlen" : cutlen,
            "epoch" : epoch,
            "name" : name,
            "heatmap" : heatmap,
        }
        dim,kernel,stride = cnn_params.values()

        self.conv = nn.Sequential(
            nn.Conv1d(1,dim,kernel_size=kernel, padding=3, stride=stride),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        )
        #Model Architecture
        if bidirect:
            self.rnn = nn.GRU(input_size = dim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            hiddenDim = hiddenDim*2
        else:
            self.rnn = nn.GRU(input_size = dim,
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
        x = self.conv(x.unsqueeze(1))
        output, hn =  self.rnn(torch.transpose(x,1,2),hidden0)
        y_hat = self.fc(output[:,-1,])
        if text == "test":
            self.cluster = torch.vstack((self.cluster,output[:,-1,].detach().clone()))
        return y_hat
