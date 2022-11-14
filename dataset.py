import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

class FormatDataset(torch.utils.data.Dataset):
      def __init__(self, pFile, nFile,dim,length,num_class):
            
            inputLen = length 
            inputDim = dim
            p = torch.load(pFile)
            n = torch.load(nFile)
            data = torch.cat((p, n)).to(torch.float)
            self.data = data.view(-1,inputDim,inputLen)
            #target 0, nontarget 1
            p_labels = torch.zeros(p.shape[0])
            n_labels = torch.ones(n.shape[0])
            labels = (torch.cat((p_labels,n_labels),dim=0).clone().detach()).to(torch.int64)
            self.label = F.one_hot(labels,num_classes=num_class).to(torch.float32)
      
      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y

class NormalDataset(torch.utils.data.Dataset):
      def __init__(self, pFile, nFile,num_class):
            
            p = torch.load(pFile)
            n = torch.load(nFile)
            self.data = torch.cat((p, n)).to(torch.float)
            #target 0, nontarget 1
            p_labels = torch.zeros(p.shape[0])
            n_labels = torch.ones(n.shape[0])
            labels = (torch.cat((p_labels,n_labels),dim=0).clone().detach()).to(torch.int64)
            self.label = F.one_hot(labels,num_classes=num_class).to(torch.float32)
      
      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y


def triple_data(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor):
      return torch.cat((a,b,c))

def triple_labels(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      return (torch.cat((a_labels,b_labels,c_labels),dim=0).clone().detach()).to(torch.int64)

def quad_data(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,d:torch.Tensor):
      return torch.cat((a,b,c,d))

def quad_labels(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,d:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      d_labels = torch.ones(d.shape[0])*3
      return (torch.cat((a_labels,b_labels,c_labels,d_labels),dim=0).clone().detach()).to(torch.int64)

class MultiDataset(torch.utils.data.Dataset):
      def __init__(self, data:dict,num_class):

            if num_class == 3:
                  self.data = triple_data(**data)
                  labels = triple_labels(**data)
            
            elif num_class == 4:
                  self.data = quad_data(**data)
                  labels = quad_labels(**data)
            
            self.label = F.one_hot(labels,num_classes=num_class).to(torch.float32)
      
      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y
      
