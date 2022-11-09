import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from unicodedata import bidirectional
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, MetricCollection, Precision,Recall
import torch.nn.functional as F
import math
### TORCH METRICS ####
def get_full_metrics(
    prefix=None,
    ):
    return MetricCollection(
        [
            Accuracy(),
            Precision(),
            Recall(),
        ],
        prefix=prefix
    )
def conv3(in_channel, out_channel, stride=1, padding=1, groups=1):
    return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride,
				   padding=padding, bias=False, dilation=padding, groups=groups)

def conv1(in_channel, out_channel, stride=1, padding=0):
    return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride,padding=padding, bias=False)

def bcnorm(channel):
    return nn.BatchNorm1d(channel)


class Bottleneck(nn.Module):
	expansion = 1.5
	def __init__(self, in_channel, out_channel, stride=1, downsample=None):
		super(Bottleneck, self).__init__()

		self.conv1 = conv1(in_channel, in_channel)
		self.bn1 = bcnorm(in_channel)

		self.conv2 = conv3(in_channel, in_channel, stride)
		self.bn2 = bcnorm(in_channel)

		self.conv3 = conv1(in_channel, out_channel)
		self.bn3 = bcnorm(out_channel)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(pl.LightningModule):
    def __init__(self, block, layers, cutlen,num_classes=2,lr=0.01):
        super(ResNet, self).__init__()
        self.chan1 = 20

		# first block
        self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
        self.bn1 = bcnorm(self.chan1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(67 , 2)

        self.lr = lr
        self.classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.cutlen = cutlen

        self.layer1 = self._make_layer(block, 20, layers[0])
        self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 67, layers[3], stride=2)

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_metrics = get_full_metrics(
            prefix="test_"
        )
        self.save_hyperparameters()


		# initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.chan1 != channels:
            downsample = nn.Sequential(
            conv1(self.chan1, channels, stride),
            bcnorm(channels),
        )

        layers = []
        layers.append(block(self.chan1, channels, stride, downsample))

        if stride != 1 or self.chan1 != channels:
            self.chan1 = channels
        for _ in range(1, blocks):
            layers.append(block(self.chan1, channels))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)

        # Logging to TensorBoard by default
        self.log("train_loss",loss)
        yhat_for_metrics = F.softmax(y_hat,dim=1)
        self.train_acc(yhat_for_metrics,y.to(torch.int64))
        self.log(
            self.train_acc,
            prog_bar=True,
            logger=True,
            on_epoch=False,
            on_step=True,
        )
        return loss

    def training_epoch_end(self, outputs):
        self.train_acc.reset()


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        self.log("valid_loss",loss)

        yhat_for_metrics = F.softmax(y_hat,dim=1)
        self.valid_acc.update(yhat_for_metrics,y.to(torch.int64))
        self.log(
            self.valid_acc,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return {"valid_loss" : loss}
    
    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.valid_acc.compute())
        self.valid_acc.reset()

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        self.log("test_loss",loss)
        yhat_for_metrics = F.softmax(y_hat,dim=1)
        self.test_metrics(yhat_for_metrics,y.to(torch.int64))
        self.log(
            self.test_metrics,
            prog_bar=True,
            logger=True,
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

