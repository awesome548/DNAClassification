import torch

def double_data(a:torch.Tensor,b:torch.Tensor,isFormat:bool,dim:int,length:int):
      return torch.cat((a,b))

def double_labels(a:torch.Tensor,b:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      return (torch.cat((a_labels,b_labels),dim=0).clone().detach()).to(torch.int64)

def triple_data(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,isFormat:bool,dim:int,length:int):
      return torch.cat((a,b,c))

def triple_labels(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      return (torch.cat((a_labels,b_labels,c_labels),dim=0).clone().detach()).to(torch.int64)

def quad_data(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,d:torch.Tensor,isFormat:bool,dim:int,length:int):
      return torch.cat((a,b,c,d))

def quad_labels(a:torch.Tensor,b:torch.Tensor,c:torch.Tensor,d:torch.Tensor):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      d_labels = torch.ones(d.shape[0])*3
      return (torch.cat((a_labels,b_labels,c_labels,d_labels),dim=0).clone().detach()).to(torch.int64)

def base_data(a,b,c,d,e,f):
      return torch.cat((a,b,c,d,e,f))

def base_labels(a,b,c,d,e,f):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      d_labels = torch.ones(d.shape[0])*3
      e_labels = torch.ones(e.shape[0])*4
      f_labels = torch.ones(f.shape[0])*5
      return (torch.cat((a_labels,b_labels,c_labels,d_labels,e_labels,f_labels),dim=0)).to(torch.int64)

class MultiDataset(torch.utils.data.Dataset):
      def __init__(self, data:list,num_classes:int,base:int):

            if num_classes == 2:
                  self.data = double_data(*data)
                  self.label = double_labels(*data)
            elif num_classes == 3:
                  self.data = triple_data(*data)
                  self.label = triple_labels(*data)
            elif num_classes == 4:
                  self.data = quad_data(*data)
                  self.label = quad_labels(*data)
            elif num_classes == base:
                  self.data = base_data(*data)
                  self.label = base_labels(*data)


      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y

