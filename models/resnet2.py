import torch
import torch.nn as nn
from process import MyProcess
import numpy as np

### TORCH METRICS ####
def conv_3(in_channel, out_channel, stride=1, padding=1, groups=1):
    return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride,
				   padding=padding, bias=False, dilation=padding, groups=groups)

def conv_1(in_channel, out_channel, stride=1, padding=0):
    return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride,padding=padding, bias=False)

def bcnorm(channel):
    return nn.BatchNorm1d(channel)

class block(nn.Module):
    def __init__(self, first_in_channels, out_channels, identity_conv=None, stride=1):
        """
        残差ブロックを作成するクラス
        Args:
            first_conv_in_channels : 1番目のconv層（1×1）のinput channel数
            first_conv_out_channels : 1番目のconv層（1×1）のoutput channel数
            identity_conv : channel数調整用のconv層
            stride : 3×3conv層におけるstide数。sizeを半分にしたいときは2に設定
        """        
        super(block, self).__init__()

        # 1番目のconv層（1×1）
        self.conv1 = conv_1(first_in_channels,out_channels)
        self.bn1 = bcnorm(out_channels)

        # 2番目のconv層（3×3）
        # パターン3の時はsizeを変更できるようにstrideは可変
        self.conv2 = conv_3(out_channels,out_channels,stride=stride)
        self.bn2 = bcnorm(out_channels)

        # 3番目のconv層（1×1）
        # output channelはinput channelの4倍になる
        self.conv3 = conv_1(out_channels,out_channels*4)
        self.bn3 = bcnorm(out_channels*4)
        self.relu = nn.ReLU()

        # identityのchannel数の調整が必要な場合はconv層（1×1）を用意、不要な場合はNone
        self.identity_conv = nn.Conv1d(out_channels) 

    def forward(self, x):

        identity = x.clone()  # 入力を保持する

        x = self.conv1(x)  # 1×1の畳み込み
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # 3×3の畳み込み（パターン3の時はstrideが2になるため、ここでsizeが半分になる）
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # 1×1の畳み込み
        x = self.bn3(x)

        # 必要な場合はconv層（1×1）を通してidentityのchannel数の調整してから足す
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x


class ResNet2(MyProcess):
    def __init__(self, block, layers, cutlen,classes,lr):
        super(ResNet, self).__init__()
        self.chan1 = 20

		# first block
        self.conv1 = nn.Conv1d(1, self.chan1, 19, padding=5, stride=3)
        self.bn1 = bcnorm(self.chan1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(67 , classes)

        self.lr = lr
        self.classes = classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.cutlen = cutlen

        self.acc = np.array([]) 
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        self.layer1 = self._make_layer(block, self.chan1, layers[0])
        self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 67, layers[3], stride=2)

        self.cluster = np.array([])        

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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer