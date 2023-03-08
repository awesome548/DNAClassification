import torch

def in_category_data_2(a,b,c,d,e,f,g):
      return torch.cat((d,e))

def in_category_label_2(a,b,c,d,e,f,g):
      y1 = torch.zeros(d.shape[0])
      y2 = torch.ones(e.shape[0])
      return (torch.cat((y1,y2),dim=0).clone().detach()).to(torch.int64)

def in_category_data(a,b,c,d,e,f,g):
      return torch.cat((b,c))

def in_category_label(a,b,c,d,e,f,g):
      y1 = torch.zeros(b.shape[0])
      y2 = torch.ones(c.shape[0])
      return (torch.cat((y1,y2),dim=0).clone().detach()).to(torch.int64)

def mix_category_data(a,b,c,d,e,f,g):
      d = d[:d.shape[0]//2,]
      return torch.cat((b,c,d))

def mix_category_label(a,b,c,d,e,f,g):
      y1 = torch.zeros(b.shape[0])
      y2 = torch.ones(c.shape[0])
      y3 = torch.ones(d.shape[0]//2)
      return (torch.cat((y1,y2,y3),dim=0).clone().detach()).to(torch.int64)

def category_data(a,b,c,d,e,f,g):
      return torch.cat((a,b,c,d,e,f,g))

def category_label(a,b,c,d,e,f,g):
      a_lbl = torch.zeros(a.shape[0])
      b_lbl = torch.ones(b.shape[0])
      c_lbl = torch.ones(c.shape[0])
      d_lbl = torch.ones(d.shape[0])*2
      e_lbl = torch.ones(e.shape[0])*2
      f_lbl = torch.ones(f.shape[0])*3
      g_lbl = torch.ones(g.shape[0])*4
      return (torch.cat((a_lbl,b_lbl,c_lbl,d_lbl,e_lbl,f_lbl,g_lbl),dim=0)).to(torch.int64)

def category_label_2(a,b,c,d,e,f,g):
      a_lbl = torch.zeros(a.shape[0])
      b_lbl = torch.ones(b.shape[0])
      c_lbl = torch.ones(c.shape[0])
      d_lbl = torch.ones(d.shape[0])
      e_lbl = torch.ones(e.shape[0])
      f_lbl = torch.ones(f.shape[0])*2
      g_lbl = torch.ones(g.shape[0])*3
      return (torch.cat((a_lbl,b_lbl,c_lbl,d_lbl,e_lbl,f_lbl,g_lbl),dim=0)).to(torch.int64)

def base_data(a,b,c,d,e,f,g):
      return torch.cat((a,b,c,d,e,f,g))

def base_labels(a,b,c,d,e,f,g):
      a_labels = torch.zeros(a.shape[0])
      b_labels = torch.ones(b.shape[0])
      c_labels = torch.ones(c.shape[0])*2
      d_labels = torch.ones(d.shape[0])*3
      e_labels = torch.ones(e.shape[0])*4
      f_labels = torch.ones(f.shape[0])*5
      g_labels = torch.ones(g.shape[0])*6
      return (torch.cat((a_labels,b_labels,c_labels,d_labels,e_labels,f_labels,g_labels),dim=0)).to(torch.int64)

class MultiDataset(torch.utils.data.Dataset):
      def __init__(self, data:list,num_classes:int):
            if num_classes == 2:
                  #self.data = in_category_data_2(*data)
                  #self.label = in_category_label_2(*data)
                  self.data = in_category_data(*data)
                  self.label = in_category_label(*data)
            #if num_classes == 2:
                  #self.data = mix_category_data(*data)
                  #self.label = mix_category_label(*data)
                  #print(self.data.shape)
                  #print(self.label.shape)
            elif num_classes == 5:
                  self.data = category_data(*data)
                  self.label = category_label(*data)
            elif num_classes == 7:
                  self.data = base_data(*data)
                  self.label = base_labels(*data)
            """
            elif num_classes == 4:
                  self.data = category_data(*data)
                  self.label = category_label_2(*data)
            """

      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y

