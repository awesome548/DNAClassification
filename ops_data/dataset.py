import torch
"""
属 genus、 科 family、目 order、綱 class、門 phylum、界 kingdom、超界 domain
"""
def family(a,b,c,d,e,f):
      a_lbl = torch.ones(a.shape[0])*0
      b_lbl = torch.ones(b.shape[0])*1
      c_lbl = torch.ones(c.shape[0])*1
      d_lbl = torch.ones(d.shape[0])*1
      e_lbl = torch.ones(e.shape[0])*1
      f_lbl = torch.ones(f.shape[0])*2
      n_class = 3
      return n_class,(torch.cat((a_lbl,b_lbl,c_lbl,d_lbl,e_lbl,f_lbl),dim=0)).to(torch.int64)
def genus(a,b,c,d,e,f):
      a_lbl = torch.ones(a.shape[0])*0
      b_lbl = torch.ones(b.shape[0])*1
      c_lbl = torch.ones(c.shape[0])*2
      d_lbl = torch.ones(d.shape[0])*3
      e_lbl = torch.ones(e.shape[0])*3
      f_lbl = torch.ones(f.shape[0])*4
      n_class = 5
      return n_class,(torch.cat((a_lbl,b_lbl,c_lbl,d_lbl,e_lbl,f_lbl),dim=0)).to(torch.int64)

def base_data(data):
      return torch.cat(data)

def base_labels(data):
      label_list = torch.zeros(1)
      for i in range(0,len(data)):
            label_list = torch.hstack((label_list,torch.ones(data[i].shape[0])*i))
      return label_list[1:].to(torch.int64)

class MultiDataset(torch.utils.data.Dataset):
      def __init__(self, data:list,mode:str):
            self.data = base_data(data)
            if mode == "genus":
                  self.classes ,self.label = genus(*data)
            elif mode == "family":
                  self.classes ,self.label = family(*data)
            else:
                  self.label = base_labels(data)

      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y

