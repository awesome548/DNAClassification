import torch
import torch.nn as nn
import numpy as np

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


class ResNet(nn.Module):
    def __init__(self, cfgs,cnnparam,mode, preference):
        super(ResNet, self).__init__()

        ### PARAMS ###
        self.lr = preference["lr"]
        classes = preference["classes"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.pref = preference
        self.cfgs = cfgs
        self.mode = mode
        self.start_time = 0
        self.end_time = 0
        self.acc = np.array([]) 
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        ######

		# first block
        c,k,s,p = cnnparam.values()
        self.chan1 = c 
        self.conv1 = nn.Conv1d(1, self.chan1,k, padding=p, stride=s)
        self.bn1 = bcnorm(self.chan1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)
        
        block = Bottleneck
        self.layer1 = self._make_layer(block, cfgs[0][0], cfgs[0][1])
        layers = preference['layers']
        self.layers = layers
        if layers > 1:
            self.layer2 = self._make_layer(block, cfgs[1][0], cfgs[1][1], stride=2)
        if layers > 2:
            self.layer3 = self._make_layer(block, cfgs[2][0], cfgs[2][1], stride=2)
        if layers > 3:
            self.layer4 = self._make_layer(block, cfgs[3][0], cfgs[3][1], stride=2)
        if layers > 4:
            self.layer5 = self._make_layer(block, cfgs[4][0], cfgs[4][1], stride=2)

        output_channel = cfgs[layers-1][0]
        self.cluster = torch.zeros(1,output_channel).cuda()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(output_channel , classes)

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
        # print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.mode == 0:
            x = self.pool(x)

        x = self.layer1(x)
        # print(x.size())
        if self.layers > 1:
            x = self.layer2(x)
            # print(x.size())
        if self.layers > 2:
            x = self.layer3(x)
            # print(x.size())
        if self.layers > 3:
            x = self.layer4(x)
            # print(x.size())
        if self.layers > 4:
            x = self.layer5(x)
            # print(x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if text == "test":
            self.cluster = torch.vstack((self.cluster,x.clone().detach()))
        x = self.fc(x)

        return x

BESTLAY =[
    [88,2],
    [96,2],
    [103,2],
    [48,2]
]
DEFAULT =[
    [20,2],
    [30,2],
    [45,2],
    [67,2],
    [101,2]
]

BESTLAY1 =[
    [80,4],
    [42,2],
    [109,1],
    [21,3],
]

DEFAULTCNN = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 3,
    "padd" : 5,
}
STRIDE12 = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 12,
    "padd" : 5,
}
STRIDE10 = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 10,
    "padd" : 5,
}
STRIDE9 = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 9,
    "padd" : 5,
}
STRIDE8 = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 8,
    "padd" : 5,
}
STRIDE5 = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 5,
    "padd" : 5,
}
STRIDE2 = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 2,
    "padd" : 5,
}
STRIDE1 = {
    "channel" : 20,
    "kernel" : 19,
    "stride" : 1,
    "padd" : 5,
}
BESTCNN = {
    "channel" : 121,
    "kernel" : 19,
    "stride" : 4,
    "padd" : 5,
}
def resnet(preference,cnnparam=DEFAULTCNN,mode=0,cfgs=DEFAULT):
    """
    c : channels
    n : num of layers
    """
    if mode == 0:
        cnnparam = STRIDE12
    elif mode == 1:
        cnnparam = STRIDE9
    elif mode == 2:
        cnnparam = STRIDE2
    elif mode == 3:
        cnnparam = STRIDE1
    elif mode == 4:
        cnnparam = STRIDE8
    print(cnnparam)
    print(f'output channel :{cfgs[preference["layers"]-1][0]}')
    return ResNet(cfgs, cnnparam,mode,preference)