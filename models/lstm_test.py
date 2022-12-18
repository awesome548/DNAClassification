
from unicodedata import bidirectional
import torch.nn as nn
import torch
from process import MyProcess

class CNNLstmEncoder(MyProcess):

    def __init__(self,hiddenDim,lr,classes,bidirect):
        super(CNNLstmEncoder,self).__init__()

        #kernel -> samples/base *2
        #stride -> samples/base

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        """
        ResNet conv
        """
        self.convDim = 20 
        self.conv = nn.Sequential(
            nn.Conv1d(1, self.convDim,kernel_size=19, padding=5, stride=3),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        )
        #Model Architecture
        if bidirect:
            self.lstm = nn.LSTM(input_size = self.convDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            self.label = nn.Linear(hiddenDim*2, classes)
        else:
            self.lstm = nn.LSTM(input_size = self.convDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            )
            self.label = nn.Linear(hiddenDim, classes)
        self.save_hyperparameters()

    def forward(self, inputs,hidden0=None):
        """
        x [batch_size , convDim , poolLen]
        """
        x = self.conv(inputs.unsqueeze(1))
        """
        x [batch_size , poolLen , convDim]
        """
        output, (hidden,cell) = self.lstm(torch.transpose(x,1,2),hidden0)
        y_hat = self.label(output[:,-1,])
        y_hat = y_hat.to(torch.float32)
        return y_hat
