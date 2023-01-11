import torch
import torch.nn as nn
from process import MyProcess
import numpy as np

### TORCH METRICS ####
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


class ResNet(MyProcess):
#class ResNet(nn.Module):
    def __init__(self, cfgs,cnnparam,mode, preference):
        super(ResNet, self).__init__()
        self.lr = preference["lr"]
        classes = preference["classes"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.pref = preference
        self.cfgs = cfgs
        self.mode = mode

		# first block
        c,k,s,p = cnnparam.values()
        self.chan1 = c 
        self.conv1 = nn.Conv1d(1, self.chan1,k, padding=p, stride=s)
        self.bn1 = bcnorm(self.chan1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)
        
        block = Bottleneck
        self.layer1 = self._make_layer(block, cfgs[0][0], cfgs[0][1])
        self.layer2 = self._make_layer(block, cfgs[1][0], cfgs[1][1], stride=2)
        self.layer3 = self._make_layer(block, cfgs[2][0], cfgs[2][1], stride=2)
        self.layer4 = self._make_layer(block, cfgs[3][0], cfgs[3][1], stride=2)
        
        output_channel = cfgs[-1][0]
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(output_channel , classes)
        self.acc = np.array([]) 
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        self.labels = torch.zeros(1).cuda()
        self.cluster = torch.zeros(1,output_channel).cuda()

        self.save_hyperparameters()

        """
        """

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

    def forward(self, x,text="train"):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.mode == 0:
            x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if text == "test":
            self.cluster = torch.vstack((self.cluster,x.clone().detach()))
        x = self.fc(x)

        return x

CFGS =[
    [20,2],
    [30,2],
    [45,2],
    [67,2]
]

def resnet(preference,cnnparam,mode=0,cfgs=CFGS):
    """
    c : channels
    n : num of layers
    """
    return ResNet(cfgs, cnnparam,mode,preference)