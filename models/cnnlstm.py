import torch.nn as nn
import torch
from process import MyProcess 
import numpy as np

class CNNLstmEncoder(MyProcess):

    def __init__(self,inputDim,outputDim,hiddenDim,lr,classes,bidirect):
        super(CNNLstmEncoder,self).__init__()

        #kernel -> samples/base *2
        #stride -> samples/base

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.classes = classes

        """
        ResNet conv
        """
        convDim = 20
        self.convDim =convDim
        self.hiddenDim = hiddenDim
        self.conv = nn.Sequential(
            nn.Conv1d(inputDim,convDim,kernel_size=19, padding=5, stride=3),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        )
        #Model Architecture
        if bidirect:
            self.lstm = nn.LSTM(input_size = convDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            self.fc = nn.Linear(hiddenDim*2, outputDim)
        else:
            self.lstm = nn.LSTM(input_size = convDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            )
            self.fc = nn.Linear(hiddenDim, outputDim)

        self.acc = np.array([])
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        #self.labels = np.array([]) 
        self.labels = torch.zeros(1,1)
        #self.position = np.array([])
        #self.cluster = np.array([])
        self.cluster = torch.zeros(1,hiddenDim*2)
        self.save_hyperparameters()

    def forward(self, inputs,hidden0=None,text="train"):
        """
        x [batch_size , convDim , poolLen]
        """
        x = self.conv(inputs)
        """
        x [batch_size , poolLen , convDim]
        """
        output, (hidden,cell) = self.lstm(torch.transpose(x,1,2),hidden0)
        if text == "test":
            last_hidden = output[:,-1,]
            #cluster = last_hidden.cpu().detach().numpy().copy()
            #self.cluster = np.append(self.cluster,cluster)
            self.cluster = torch.vstack((self.cluster,last_hidden.detach()))
        y_hat = self.fc(output[:,-1,])
        return y_hat


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer
