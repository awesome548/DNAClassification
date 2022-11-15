import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

def double_data(a:torch.Tensor,b:torch.Tensor,isFormat:bool,dim:int,length:int):
      if isFormat:
            return (torch.cat((a,b))).view(-1,dim,length)
      else:
            return torch.cat((a,b))

def double_labels(a:torch.Tensor,b:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      return (torch.cat((a_labels,b_labels),dim=0).clone().detach()).to(torch.int64)

def triple_data(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,isFormat:bool,dim:int,length:int):
      if isFormat:
            return (torch.cat((a,b,c))).view(-1,dim,length)
      else:
            return torch.cat((a,b,c))

def triple_labels(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      return (torch.cat((a_labels,b_labels,c_labels),dim=0).clone().detach()).to(torch.int64)

def quad_data(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,d:torch.Tensor,isFormat:bool,dim:int,length:int):
      if isFormat:
            return (torch.cat((a,b,c,d))).view(-1,dim,length)
      else:
            return torch.cat((a,b,c,d))

def quad_labels(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,d:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      d_labels = torch.ones(d.shape[0])*3
      return (torch.cat((a_labels,b_labels,c_labels,d_labels),dim=0).clone().detach()).to(torch.int64)

class MultiDataset(torch.utils.data.Dataset):
      def __init__(self, data:dict,num_class:int,transform:dict):

            if num_class == 2:
                  self.data = double_data(**data,**transform)
                  labels = double_labels(**data)
            elif num_class == 3:
                  self.data = triple_data(**data,**transform)
                  labels = triple_labels(**data)
            elif num_class == 4:
                  self.data = quad_data(**data,**transform)
                  labels = quad_labels(**data)
            
            self.label = F.one_hot(labels,num_classes=num_class).to(torch.float32)
      
      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y
      
