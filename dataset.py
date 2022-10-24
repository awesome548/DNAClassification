import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
  def __init__(self, pFile, nFile,size):
        p = torch.load(pFile)
        n = torch.load(nFile)
        data = torch.cat((p, n)).to(torch.float)
        #print(data.size())
        self.data = data.view(-1,size[0],size[1])
        #target 0, nontarget 1
        p_labels = torch.zeros(p.shape[0])
        n_labels = torch.ones(n.shape[0])
        labels = torch.tensor(torch.cat((p_labels,n_labels),dim=0)).to(torch.int64)
        self.label = F.one_hot(labels,num_classes=2).to(torch.float32)

  def __len__(self):
        return len(self.label)

  def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]
        return X, y
