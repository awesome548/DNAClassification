import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class MyProcess(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y)

        # Logging to TensorBoard by default
        self.log("train_loss",loss)
        return loss


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        
        self.log("valid_loss",loss)
        return {"valid_loss" : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.log("avg_val__loss",avg_loss)
        return {"avg_val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_float =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y_float)
        
        y_hat_index = y_hat.max(dim=1).indices
        self.log("test_loss",loss)
        self.log_dict(self.metrics(y_hat_index,y))
        return {"test_loss" : loss}
