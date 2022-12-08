import torch.nn as nn
import torch
import math
from process import MyProcess
from einops import rearrange,repeat
from torchmetrics.functional.classification import multilabel_accuracy

"""
not recommended
pos_emb : sin&cos
cls_token : false
"""

def positionalencoding1d(x,dtype = torch.float32):
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
    def __init__(self,classes,head_num,lr,block_num=6,dim_head=64,mlp_dim=2048):
        super(SimpleViT,self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        self.convDim = 20
        self.conv = nn.Sequential(
            nn.Conv1d(1,20,19, padding=5, stride=3),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        )

        self.transformer = Transformer(self.convDim,block_num,head_num,dim_head,mlp_dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(self.convDim),
            nn.Linear(self.convDim,classes)
        )

        self.classes = classes
        # Metrics
        self.save_hyperparameters()


    def forward(self, inputs):
        b,_,_ = inputs.shape
        x = self.conv(inputs)
        x = torch.transpose(x,1,2)
        ### x = [b, len , dim]

        pe = positionalencoding1d(x)
        pe = repeat(pe,'l d -> b l d',b=b)
        x = x + pe

        x = self.transformer(x)
        x_ = x.mean(dim = 1)
        x__ = self.to_latent(x_)
        return self.linear_head(x__)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer


