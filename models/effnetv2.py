import numpy as np
import torch
import torch.nn as nn
import math
from process import MyProcess

#__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c,  _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


def conv_3x3_bn(inp, oup, kernel,stride,padd):
    return nn.Sequential(
        nn.Conv1d(inp, oup, kernel, stride, padd, bias=False),
        #nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm1d(oup),
        SiLU(),
        nn.MaxPool1d(2, padding=1, stride=2),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm1d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv1d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(MyProcess):
    def __init__(self, cfgs,preference, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.lr = preference["lr"]
        classes = preference["classes"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.pref = preference
        conv_param = cfgs.pop(-1)
        assert len(cfgs) == 6
        self.cfgs = cfgs
        self.conv = conv_param

        # building first layer
        #input_channel = _make_divisible(24 * width_mult, 8)
        input_channel = conv_param[0]
        #layers = [conv_3x3_bn(3, input_channel, 2)]
        layers = [conv_3x3_bn(1,*conv_param)]

        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(output_channel, classes)
        
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
        self._initialize_weights()

    def forward(self, x,text="train"):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if text == "test":
            self.cluster = torch.vstack((self.cluster,x.clone().detach()))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

CFGS =[
    # t, c, n, s, SE
    [1,  24,  2, 1, 0],
    [4,  48,  4, 2, 0],
    [4,  64,  4, 2, 0],
    [4, 128,  4, 2, 1],
    [6, 160,  4, 1, 1],
    [6, 256,  4, 2, 1],
]

def effnetv2_s(preference,cfgs=CFGS,**kwargs):
    """
    Constructs a EfficientNetV2-S model
    t : expand ratio
    c : channels
    n : num of layers
    s : stride of conv
    """
    return EffNetV2(cfgs, preference, **kwargs)