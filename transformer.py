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

class PositionalEncoding(nn.Module):
    def __init__(self,max_len: int,d_model: int , dropout: float =0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,dim,head_num=2,dropout=0.1) -> None:
        super(MultiHeadAttention,self).__init__()
        self.dim = dim
        self.head_num = head_num


        d_head = int(dim/head_num)
        self.linear_Q = nn.Linear(dim, dim, bias=False)
        self.linear_K = nn.Linear(dim, dim, bias=False)
        self.linear_V = nn.Linear(dim, dim, bias=False)
        self.linear = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def split_head(self, x):
        x = torch.tensor_split(x, self.head_num, dim = 2)
        x = torch.stack(x, dim = 1)
        return x

    def concat_head(self, x):
        x = torch.tensor_split(x, x.size()[1], dim = 1)
        x = torch.concat(x, dim = 3).squeeze(dim = 1)
        return x

    def forward(self, Q, K, V, mask = None):
        Q = self.linear_Q(Q)   #(BATCH_SIZE,word_count,dim)
        K = self.linear_K(K)
        V = self.linear_V(V)

        Q = self.split_head(Q)   #(BATCH_SIZE,head_num,word_count//head_num,dim)
        K = self.split_head(K)
        V = self.split_head(V)

        QK = torch.matmul(Q, torch.transpose(K, 3, 2))
        QK = QK/((self.dim//self.head_num)**0.5)

        if mask is not None:
            QK = QK + mask

        softmax_QK = self.softmax(QK)
        softmax_QK = self.dropout(softmax_QK)

        QKV = torch.matmul(softmax_QK, V)
        QKV = self.concat_head(QKV)
        QKV = self.linear(QKV)
        return QKV

class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0.1,mlp_ratio=4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(dim, dim*mlp_ratio)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(dim*mlp_ratio, dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, head_num, dropout = 0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(dim, head_num)
        self.layer_norm_1 = nn.LayerNorm([dim])
        self.layer_norm_2 = nn.LayerNorm([dim])
        self.FF = FeedForward(dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        Q = K = V = x
        x = self.MHA(Q, K, V)
        x = self.dropout_1(x)
        x = x + Q
        x = self.layer_norm_1(x)
        _x = x
        x = self.FF(x)
        x = self.dropout_2(x)
        x = x + _x
        x = self.layer_norm_2(x)
        return x


class ViTransformer(MyProcess):
    def __init__(self,classes,length,dropout=0.1,head_num=2,block_num=6,lr=0.001):
        super(ViTransformer,self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        self.inputDim = 1
        self.block_num = block_num
        self.convDim = 20

        self.poolLen = ((length+5*2-19)//3 + 1)//2 + 1

        self.conv = nn.Sequential(
            nn.Conv1d(1,20,19, padding=5, stride=3),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        )
        self.class_token = nn.Parameter(torch.rand(1,self.convDim))
        # Class token added 
        self.PE = PositionalEncoding(self.poolLen+1,self.convDim)
        self.dropout = nn.Dropout(dropout)
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(self.convDim, head_num) for _ in range(block_num)])
        self.Classification = nn.Linear(self.convDim,classes)
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
        tokens = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])
        x = self.PE(tokens)
        x = self.dropout(x)

        for i in range(self.block_num):
            x = self.EncoderBlocks[i](x)

        x = x[:,0]
        return self.Classification(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer


