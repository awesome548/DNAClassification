from unicodedata import bidirectional
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, MetricCollection, Precision,Recall
import torch.nn.functional as F
import torch
import math

### TORCH METRICS ####
def get_full_metrics(
    average_method="macro",
    classes=None,
    prefix=None,
    ):
    return MetricCollection(
        [
            Accuracy(),
            Precision(
                average=average_method,
                num_classes=classes,
            ),
            Recall(
                average=average_method,
                num_classes=classes,
            ),
        ],
        prefix= prefix
    )

def part_metrics(
    prefix=None,
    ignore_index=None,
    ):
    return MetricCollection(
        [
            Accuracy(
                ignore_index=ignore_index,
            ),
        ],
        prefix=prefix
    )

### CREATE MODEL ###
class LstmEncoder(pl.LightningModule):
    def __init__(self,inputDim,outputDim,hiddenDim,lr,classes,bidirect):
        super(LstmEncoder,self).__init__()

        self.lr = lr
        self.classes = classes
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
        self.train_metrics = part_metrics(prefix="train_")
        self.valid_metrics = part_metrics(prefix="valid_")
        self.test_metrics = get_full_metrics(
            num_classes=classes,
            prefix="test_"
        )
        self.save_hyperparameters()

    def forward(self, inputs,hidden0=None):
        # in lightning, forward defines the prediction/inference actions
        output, (hidden,cell) = self.lstm(inputs,hidden0)
        y_hat = self.label(output[:,-1,])
        y_hat = y_hat.to(torch.float32)
        return y_hat

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)

        # Logging to TensorBoard by default
        y_hat = F.softmax(y_hat,dim=1)
        y = y.to(torch.int64)

        self.log("train_loss",loss)
        self.log("train_acc",self.train_metrics(y_hat,y))
        return loss


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        
        y_hat = F.softmax(y_hat,dim=1)
        y = y.to(torch.int64)
        self.log("valid_loss",loss)
        self.log("valid_acc",self.valid_metrics(y_hat,y))
        return {"valid_loss" : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.log("avg_val__loss",avg_loss)
        return {"avg_val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        
        
        y_hat = F.softmax(y_hat,dim=1)
        y = y.to(torch.int64)
        self.log("test_loss",loss)
        self.log_dict(
            self.test_metrics(y_hat,y),
            on_epoch=True,
        )
        return {"test_loss" : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer

class CNNLstmEncoder(pl.LightningModule):

    def __init__(self,inputDim,outputDim,hiddenDim,lr,classes,bidirect,padd,ker,stride,convDim):
        super(CNNLstmEncoder,self).__init__()

        #kernel -> samples/base *2
        #stride -> samples/base

        self.lr = lr
        self.classes = classes
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

        self.train_metrics = part_metrics(prefix="train_")
        self.valid_metrics = part_metrics(prefix="valid_")
        self.test_metrics = get_full_metrics(
            num_classes=classes,
            prefix="test_"
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

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)

        # Logging to TensorBoard by default
        y_hat = F.softmax(y_hat,dim=1)
        y = y.to(torch.int64)
        self.log("train_loss",loss)
        self.log(self.train_metrics(y_hat,y))
        return loss


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)
        
        y_hat = F.softmax(y_hat,dim=1)
        y = y.to(torch.int64)
        self.log("valid_loss",loss)
        self.log(self.valid_metrics(y_hat,y))
        return {"valid_loss" : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.log("avg_val__loss",avg_loss)
        return {"avg_val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)
        
        y_hat = F.softmax(y_hat,dim=1)
        y = y.to(torch.int64)
        self.log("test_loss",loss)
        self.test_metrics(y_hat,y.to(torch.int64))
        self.log_dict(
            self.test_metrics,
            on_epoch=True,
            on_step=False,
        )
        return {"test_loss" : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer
