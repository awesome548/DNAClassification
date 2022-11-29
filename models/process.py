import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np


class MyProcess(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y =  F.one_hot(y,num_classes=self.classes).to(torch.float32)
        loss = self.loss_fn(y_hat,y)

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

        self.log("test_loss",loss)
        y_hat_idx = y_hat.max(dim=1).indices
        acc = (y == y_hat_idx).float().mean().item()
        y_hat_idx = y_hat_idx.cpu().detach().numpy().copy()
        y = y.cpu().detach().numpy().copy()
        y_hat_idx = (y_hat_idx == 0)
        y = (y == 0)
        tp = np.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
        fp = np.count_nonzero((y_hat_idx == True) & (y_hat_idx != y))
        tn = np.count_nonzero((y_hat_idx == False) & (y_hat_idx == y))
        fn = np.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))

        self.metric['tp'] += tp
        self.metric['fp'] += fp
        self.metric['tn'] += tn
        self.metric['fn'] += fn

        self.acc = np.append(self.acc,acc)
        return {"test_loss" : loss}

    def test_epoch_end(self,outputs) -> None:
        tp,fp,fn,tn = self.metric.values()
        acc = self.acc.mean()
        acc_1 = (tp+tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)

        self.log("test_Accuracy",acc)
        self.log("test_Accuracy2",acc_1)
        self.log("test_Recall",recall)
        self.log("test_Precision",precision)
        return super().test_epoch_end(outputs)
