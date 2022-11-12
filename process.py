import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class MyProcess(pl.LightningModule):
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
        self.train_metrics(y_hat,y)
        self.log_dict(self.train_metrics,on_epoch=False,on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat,y)
        
        y_hat = F.softmax(y_hat,dim=1)
        y = y.to(torch.int64)
        self.log("valid_loss",loss)
        self.valid_metrics(y_hat,y)
        self.log_dict(self.valid_metrics,on_epoch=False,on_step=True)
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
        self.test_metrics(y_hat,y)
        self.log_dict(self.test_metrics,on_epoch=True,on_step=False)
        return {"test_loss" : loss}
