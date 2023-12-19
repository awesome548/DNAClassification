import torch
from torch import nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

"""
pos_emb : cos&sin
cls_token : false
"""

def posemb_sincos_1d(x,dtype = torch.float32):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    _,l,d,device,dtype= *x.shape, x.device,x.dtype
    if d % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d))
    pe = torch.zeros(l, d,device=device)
    position = torch.arange(0, l).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d, 2, dtype=torch.float) * -(math.log(10000.0) / d)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.type(dtype)
# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT2(nn.Module):
    def __init__(self, classes, depth, heads, lr,mlp_dim=1024, dim_head = 64):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.classes = classes

        dim = 20

        self.conv = nn.Sequential(
            nn.Conv1d(1,dim,19, padding=5, stride=3),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        ) 
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes)
        )
        self.acc = np.array([])   
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }       # Metrics
        self.save_hyperparameters()

    def forward(self, inputs):
        x.unsqueeze(1)
        x = self.conv(inputs)
        x = torch.transpose(x,1,2)
        pe = posemb_sincos_1d(x)
        x = x + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer