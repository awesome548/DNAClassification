import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, MetricCollection, Precision,Recall
import torch.nn.functional as F
import torch

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

class LstmEncoder(pl.LightningModule):

    def __init__(self,inputDim,hiddenDim,outputDim,lr,classes=2):
        super(LstmEncoder,self).__init__()

        self.lr = lr
        self.classes = classes
        self.loss_fn = nn.MSELoss()

        #Model Architecture
        self.lstm = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.label = nn.Linear(hiddenDim, outputDim)
        self.train_metrics = get_full_metrics(
            num_classes=classes,
            prefix="train_",
        )
        self.valid_metrics = get_full_metrics(
            num_classes=classes,
            prefix="valid_"
        )

    def forward(self, inputs,hidden0=None):
        # in lightning, forward defines the prediction/inference actions
        output, (hidden,cell) = self.lstm(inputs,hidden0)
        y_hat = self.label(output[:,])
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
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        yhat_for_metrics = F.softmax(y_hat,dim=1)
        self.train_metrics(yhat_for_metrics,y.to(torch.int64))
        self.log_dict(
            self.train_metrics,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=True,            
        )
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer

    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        yhat_for_metrics = F.softmax(y_hat,dim=1)
        self.valid_metrics(yhat_for_metrics,y.to(torch.int64))
        self.log_dict(
            self.valid_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,            
        )
        return {"valid_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}