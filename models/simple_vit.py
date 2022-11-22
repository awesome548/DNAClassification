from unicodedata import bidirectional
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, MetricCollection, Precision,Recall
import torch.nn.functional as F
import torch
import math
from metrics import get_full_metrics
from process import MyProcess
from einops import rearrange
from einops.layers.torch import Rearrange

class PositionalEncoding(nn.Module):
    def __init__(self,max_len: int,d_model: int):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, max_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head = 64):
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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


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

class SimpleViT(MyProcess):
    def __init__(self,classes,length,head_num,lr,block_num=6,dim_head=64,mlp_dim=2048):
        super(SimpleViT,self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.classes = classes

        self.convDim = 20

        self.poolLen = ((length+5*2-19)//3 + 1)//2 + 1

        self.conv = nn.Sequential(
            nn.Conv1d(1,20,19, padding=5, stride=3),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        )
        self.PE = PositionalEncoding(self.poolLen,self.convDim)

        self.transformer = Transformer(self.convDim,block_num,head_num,dim_head,mlp_dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(self.convDim),
            nn.Linear(self.convDim,classes)
        )



        # Metrics
        self.train_metrics = get_full_metrics(
            num_classes=classes,
            prefix="train_",
        )
        self.valid_metrics = get_full_metrics(
            num_classes=classes,
            prefix="valid_",
        )
        self.test_metrics = get_full_metrics(
            num_classes=classes,
            prefix="test_",
        )
        self.save_hyperparameters()


    def forward(self, inputs):
        x = self.conv(inputs)
        x = torch.transpose(x,1,2)
        ### x = [b, len , dim]
        x = self.PE(x)

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


