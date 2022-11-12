from unicodedata import bidirectional
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, MetricCollection, Recall,ConfusionMatrix,Precision
import torch.nn.functional as F
import torch
import math
from process import MyProcess

def get_full_metrics(
    threshold=0.5,
    average_method="macro",
    num_classes=None,
    prefix=None,
    ignore_index=None,
):
    return MetricCollection(
        [
            Accuracy(
                threshold=threshold,
                ignore_index=ignore_index,
            ),
            Precision(
                threshold=threshold,
                average=average_method,
                num_classes=num_classes,
                ignore_index=ignore_index,
            ),
            Recall(
                threshold=threshold,
                average=average_method,
                num_classes=num_classes,
                ignore_index=ignore_index,
            ),
        ],
        prefix= prefix
    )

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

    def __init__(self,inputDim,outputDim,hiddenDim,lr,classes,bidirect,padd,ker,stride,convDim):
        super(CNNLstmEncoder,self).__init__()

        #kernel -> samples/base *2
        #stride -> samples/base

        self.lr = lr
        self.loss_fn = nn.MSELoss()

        """
        ResNet conv
        """
        self.convDim = convDim
        self.conv = nn.Sequential(
            nn.Conv1d(inputDim, self.convDim,kernel_size=ker, padding=padd, stride=stride),
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
            self.label = nn.Linear(hiddenDim*2, outputDim)
        else:
            self.lstm = nn.LSTM(input_size = self.convDim,
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
