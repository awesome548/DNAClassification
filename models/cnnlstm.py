import torch.nn as nn
import torch
from models.process import MyProcess
from models.metrics import get_full_metrics
import numpy as np

### CREATE MODEL ###
class LstmEncoder(MyProcess):
    def __init__(self,inputDim,outputDim,hiddenDim,lr,classes,bidirect):
        super(LstmEncoder,self).__init__()

        self.lr = lr
        self.loss_fn = nn.MSELoss()

        #Model Architecture
        if bidirect:
            self.lstm = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            self.label = nn.Linear(hiddenDim*2, outputDim)
        else:
            self.lstm = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            )
            self.label = nn.Linear(hiddenDim, outputDim)
        self.train_metrics = get_full_metrics(
            num_classes=classes,
            prefix="train_",
        )
        self.valid_metrics = get_full_metrics(
            num_classes=classes,
            prefix="valid_",
        )
        self.test_metrics = get_full_metrics(
            num_classes=classes,
            prefix="test_",
        )
        self.save_hyperparameters()

    def forward(self, inputs,hidden0=None):
        # in lightning, forward defines the prediction/inference actions
        output, (hidden,cell) = self.lstm(inputs,hidden0)
        y_hat = self.label(output[:,-1,])
        y_hat = y_hat.to(torch.float32)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer

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
            self.label = nn.Linear(hiddenDim*2, outputDim)
        else:
            self.lstm = nn.LSTM(input_size = convDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            )
            self.label = nn.Linear(hiddenDim, outputDim)

        self.acc = np.array([])
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        self.metrics = get_full_metrics(classes=classes,prefix="Test_")
        self.save_hyperparameters()

    def forward(self, inputs,hidden0=None):
        """
        x [batch_size , convDim , poolLen]
        """
        x = self.conv(inputs)
        """
        x [batch_size , poolLen , convDim]
        """
        output, (hidden,cell) = self.lstm(torch.transpose(x,1,2),hidden0)
        y_hat = self.label(output[:,-1,])
        y_hat = y_hat.to(torch.float32)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer
