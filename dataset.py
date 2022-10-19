import torch
from torch.utils.data.dataset import Dataset

class Dataset(torch.utils.data.Dataset):
      def __init__(self, aFile, bFile,cFile,dFile):
            a = torch.load(aFile)
            b = torch.load(bFile)
            c = torch.load(cFile)
            d = torch.load(dFile)
            self.data = torch.cat((a,b,c,d))
            #target 0, nontarget 1
            #self.label = torch.cat((torch.zeros(z.shape[0]), torch.ones(h.shape[0]))) #human: 1, others: 0
            #Target: 0, nontarget: 1,2,3
            self.label = torch.cat((torch.zeros(a.shape[0]),torch.ones(b.shape[0]),torch.ones(c.shape[0])*2,torch.ones(d.shape[0])*3))
        

      def __len__(self):
            return len(self.label)
      
      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y
