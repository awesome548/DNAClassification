import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD

class LstmEncoder(pl.LightningModule):

    def __init__(self,inputDim,hiddenDim,outputDim,lr=0.001,classes=2):
        super(LstmEncoder,self).__init__()

        self.lr = lr
        self.classes = classes
        self.loss_fn = nn.MSELoss()

        #Model Architecture
        self.lstm = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.label = nn.Linear(hiddenDim, outputDim)

    def forward(self, inputs,hidden0=None):
        # in lightning, forward defines the prediction/inference actions
        output, (hidden,cell) = self.lstm(inputs,hidden0)
        return self.label(output[:,-1,:])

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)

        # Logging to TensorBoard by default
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer