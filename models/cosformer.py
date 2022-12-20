from models import Kernel_transformer 
from process import MyProcess
import torch
import torch.nn as nn
import numpy as np

class Transformer_clf_model(MyProcess):
    def __init__(self, model_type, model_args,lr,classes,cutlen,target):
        super(Transformer_clf_model, self).__init__()
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.classes = classes
        self.target = target
        
        dim = model_args["d_model"] 
        self.conv = nn.Sequential(
            nn.Conv1d(1,dim,kernel_size=19, padding=5, stride=3),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=2),
        )
        max_len = ((cutlen+5*2-19)//3 + 1)//2 + 2
        self.cls_token = nn.Parameter(torch.rand(1,dim))
        if model_type == 'kernel':
            self.encoder = Kernel_transformer(**model_args,max_len=max_len)
        else:
            raise NotImplementedError(
                "The only options for model_type are 'kernel' and 'baseline'.")

        self.linear = nn.Linear(model_args['d_model'], classes)
        self.acc = np.array([])
        self.metric = {
            'tp' : 0,
            'fp' : 0,
            'fn' : 0,
            'tn' : 0,
        }
        
        self.labels = torch.zeros(1).cuda()
        self.cluster = torch.zeros(1,model_args['d_model']).cuda()
        self.save_hyperparameters()

    def forward(self, x, attention_mask=None, lengths=None,text=None):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.transpose(x,1,2)
        input_tokens = torch.stack([torch.vstack((self.cls_token, x[i])) for i in range(len(x))])
        x = self.encoder(
            input_ids=input_tokens,
            attention_mask=attention_mask,
            lengths=lengths)[:, 0, :]
        # x -> [batch_size, d_model]
        if text == "test":
            self.cluster = torch.vstack((self.cluster,x.clone().detach()))

        x = self.linear(x)
        
        return x