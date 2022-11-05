from unicodedata import bidirectional
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, MetricCollection, Precision,Recall
import torch.nn.functional as F
import torch
import math

### TORCH METRICS ####
def get_full_metrics(
    threshold=0.5,
    average_method="macro",
    num_classes=None,
    prefix=None,
    ignore_index=None,
    ):
    return MetricCollection(
        [
            Accuracy(
                threshold=threshold,
                ignore_index=ignore_index,
            ),
            Precision(
                threshold=threshold,
                average=average_method,
                num_classes=num_classes,
                ignore_index=ignore_index,
            ),
            Recall(
                threshold=threshold,
                average=average_method,
                num_classes=num_classes,
                ignore_index=ignore_index,
            ),
        ],
        prefix= prefix
    )

def part_metrics(
    threshold=0.5,
    average_method="macro",
    num_classes=None,
    prefix=None,
    ignore_index=None,
    ):
    return MetricCollection(
        [
            Accuracy(
                threshold=threshold,
                ignore_index=ignore_index,
            ),
        ],
        prefix=prefix
    )

### CREATE MODEL ###
class LstmEncoder(pl.LightningModule):
    def __init__(self,inputDim,outputDim,hiddenDim,lr,classes,bidirect):
        super(LstmEncoder,self).__init__()

        self.lr = lr
        self.classes = classes
        self.loss_fn = nn.MSELoss()

        #Model Architecture
        if bidirect:
            self.lstm = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            self.label = nn.Linear(hiddenDim*2, outputDim)
        else:
            self.lstm = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            )
            self.label = nn.Linear(hiddenDim, outputDim)
        self.train_metrics = part_metrics(
            num_classes=classes,
            prefix="train_",
        )
        self.valid_metrics = part_metrics(
            num_classes=classes,
            prefix="valid_"
        )
        self.test_metrics = get_full_metrics(
            num_classes=classes,
            prefix="test_"
        )
        self.save_hyperparameters()

    def forward(self, inputs,hidden0=None):
        # in lightning, forward defines the prediction/inference actions
        output, (hidden,cell) = self.lstm(inputs,hidden0)
        y_hat = self.label(output[:,-1,])
        y_hat = y_hat.to(torch.float32)
        return y_hat

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
        self.train_metrics(yhat_for_metrics,y.to(torch.int64))
        self.log_dict(
            self.train_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=False,
            on_step=True,            
        )
        return loss


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        self.log("valid_loss",loss)
        yhat_for_metrics = F.softmax(y_hat,dim=1)
        self.valid_metrics(yhat_for_metrics,y.to(torch.int64))
        self.log_dict(
            self.valid_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,            
        )
        return {"valid_loss" : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.log("avg_val__loss",avg_loss)
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
        self.log_dict(
            self.test_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=False,
            on_step=True,            
        )
        return {"test_loss" : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer

class CNNLstmEncoder(pl.LightningModule):

    def __init__(self,inputDim,outputDim,hiddenDim,lr,classes,bidirect,padd,ker,stride,convDim):
        super(CNNLstmEncoder,self).__init__()

        #kernel -> samples/base *2 
        #stride -> samples/base 

        self.lr = lr
        self.classes = classes
        self.loss_fn = nn.MSELoss()

        self.convDim = convDim
        convLen = ((3000+2*padd-ker)/stride) + 1
        self.poolLen = int(((convLen - 2) / 2) + 1)
        #Model Architecture
        self.conv = nn.Sequential(
            nn.Conv1d(inputDim, self.convDim,kernel_size=ker, padding=padd, stride=stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0, stride=2),
        )
        #Model Architecture
        if bidirect:
            self.lstm = nn.LSTM(input_size = self.poolLen,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            bidirectional = True,
                            )
            self.label = nn.Linear(hiddenDim*2, outputDim)
        else:
            self.lstm = nn.LSTM(input_size = self.poolLen,
                            hidden_size = hiddenDim,
                            batch_first = True,
                            )
            self.label = nn.Linear(hiddenDim, outputDim)
        
        
        self.train_metrics = part_metrics(
            num_classes=classes,
            prefix="train_",
        )
        self.valid_metrics = part_metrics(
            num_classes=classes,
            prefix="valid_"
        )
        self.test_metrics = get_full_metrics(
            num_classes=classes,
            prefix="test_"
        )
        self.save_hyperparameters()

    def forward(self, inputs,hidden0=None):
        # in lightning, forward defines the prediction/inference actions
        x = self.conv(inputs)
        output, (hidden,cell) = self.lstm(x,hidden0)
        y_hat = self.label(output[:,-1,])
        y_hat = y_hat.to(torch.float32)
        return y_hat

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
        self.train_metrics(yhat_for_metrics,y.to(torch.int64))
        self.log_dict(
            self.train_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=False,
            on_step=True,            
        )
        return loss


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        self.log("valid_loss",loss)
        yhat_for_metrics = F.softmax(y_hat,dim=1)
        self.valid_metrics(yhat_for_metrics,y.to(torch.int64))
        self.log_dict(
            self.valid_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,            
        )
        return {"valid_loss" : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.log("avg_val__loss",avg_loss)
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
        self.log_dict(
            self.test_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=False,
            on_step=True,            
        )
        return {"test_loss" : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer

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

class ViTransformer(pl.LightningModule):
    def __init__(self,length,n_patches,hiddenDim,classes=2,dropout=0.1,head_num=2,block_num=2,lr=0.01):
        super(ViTransformer,self).__init__()

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        self.inputDim = length//n_patches
        self.n_patches = n_patches
        self.hiddenDim = hiddenDim
        self.block_num = block_num
        
        # 1) Linear mapper
        """
        x : [batch_size , max_len , hiddenDim]
        """
        self.linear_mapper = nn.Linear(self.inputDim,self.hiddenDim)

        # 2) learnable classification token
        """
        output ; [batch_size , 1(class token) + max_len, hiddenDim]
        """
        self.class_token = nn.Parameter(torch.rand(1,self.hiddenDim))        

        self.PE = PositionalEncoding(n_patches+1,self.hiddenDim)
        self.dropout = nn.Dropout(dropout)
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(self.hiddenDim, head_num) for _ in range(block_num)])
        self.Classification = nn.Sequential(
            nn.Linear(self.hiddenDim,classes),
            nn.Softmax(dim=-1)
        )
        # Metrics 
        self.train_metrics = part_metrics(
            num_classes=classes,
            prefix="train_",
        )
        self.valid_metrics = part_metrics(
            num_classes=classes,
            prefix="valid_"
        )
        self.test_metrics = get_full_metrics(
            num_classes=classes,
            prefix="test_"
        )
        self.save_hyperparameters()

    def forward(self, inputs):
        
        tokens = self.linear_mapper(inputs.view(-1,self.n_patches,self.inputDim))

        #adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        x = self.PE(tokens)
        x = self.dropout(x)

        for i in range(self.block_num):
            x = self.EncoderBlocks[i](x)
        
        x = x[:,0]
        return self.Classification(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)

        # Logging to TensorBoard by default
        self.log("train_loss",loss)
        self.train_metrics(y_hat,y.to(torch.int64))
        self.log_dict(
            self.train_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=False,
            on_step=True,            
        )
        return loss


    def validation_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        self.log("valid_loss",loss)
        self.valid_metrics(y_hat,y.to(torch.int64))
        self.log_dict(
            self.valid_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,            
        )
        return {"valid_loss" : loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        self.log("avg_val__loss",avg_loss)
        return {"avg_val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.to(torch.float32)
        loss = self.loss_fn(y_hat,y)
        self.log("test_loss",loss)
        self.test_metrics(y_hat,y.to(torch.int64))
        self.log_dict(
            self.test_metrics,
            prog_bar=True,
            logger=True,
            on_epoch=False,
            on_step=True,            
        )
        return {"test_loss" : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            )
        return optimizer



