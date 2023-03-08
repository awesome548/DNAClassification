from process import Preprocess,calu_size
import torch
import glob
from torch.utils.data import DataLoader

def base_class(idset: list,dataset:list,size:int,cut_size:dict) -> dict:
    cutoff,cutlen,maxlen,stride = cut_size.values()
    lam_size = 1000
    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataC = Preprocess(idset[2]).process(inpath=dataset[2],**cut_size,size=size)
    dataD = Preprocess(idset[3]).process(inpath=dataset[3],**cut_size,size=size)
    dataE = Preprocess(idset[4]).process(inpath=dataset[4],**cut_size,size=size)
    dataF = Preprocess(idset[5]).process(inpath=dataset[5],**cut_size,size=size)
    #dataG = Preprocess(idset[6]).process(inpath=dataset[6],**cut_size,size=lam_size)
    dataG = Preprocess(idset[6]).process(inpath=dataset[6],**cut_size,size=size)
    manipulate = calu_size(cutlen,maxlen,stride)
    dataset_size = manipulate*size
    lam_size = manipulate*lam_size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    lambda_size = [int(lam_size*0.8),int(lam_size*0.1),int(lam_size*0.1)]
    
    #assert dataG.shape[0] == 2000
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)
    train_C, val_C, test_C = torch.split(dataC,data_size)
    train_D, val_D, test_D = torch.split(dataD,data_size)
    train_E, val_E, test_E = torch.split(dataE,data_size)
    train_F, val_F, test_F = torch.split(dataF,data_size)
    #train_G, val_G, test_G = torch.split(dataG,lambda_size)
    train_G, val_G, test_G = torch.split(dataG,data_size)

    train = [train_A,train_B,train_C,train_D,train_E,train_F,train_G]
    val = [val_A,val_B,val_C,val_D,val_E,val_F,val_G]
    test = [test_A,test_B,test_C,test_D,test_E,test_F,test_G]
    
    return train,val,test,dataset_size

class Dataformat2:
    def __init__(self,target: list,inpath:list,dataset_size:int,cut_size:dict,num_classes:int,idx:list) -> None:
        idset = glob.glob(target+'/*.txt')
        dataset = glob.glob(inpath+'/*')
        idset.sort()
        dataset.sort()

        train, val, test, dataset_size = base_class(idset,dataset,dataset_size,cut_size)

        self.test_set = MultiDataset2(test,num_classes,idx)
        pass
    
    def test_loader(self,batch):
        params = {'batch_size': batch,
				'shuffle': False,
				'num_workers': 24}
        return DataLoader(self.test_set,**params)


def category_data(a,b,c,d,e,f,g,idx):
    idx = idx.cpu()
    return torch.cat((a,b,c,d,e,f,g))[idx,:]

def category_label(a,b,c,d,e,f,g,idx):
    idx = idx.cpu()
    a_lbl = torch.ones(a.shape[0])
    b_lbl = torch.zeros(b.shape[0])
    c_lbl = torch.ones(c.shape[0])
    d_lbl = torch.ones(d.shape[0])
    e_lbl = torch.ones(e.shape[0])
    f_lbl = torch.ones(f.shape[0])
    g_lbl = torch.ones(g.shape[0])
    return (torch.cat((a_lbl,b_lbl,c_lbl,d_lbl,e_lbl,f_lbl,g_lbl),dim=0))[idx].to(torch.int64)


class MultiDataset2(torch.utils.data.Dataset):
      def __init__(self, data:list,num_classes:int,idx:list):
        self.data = category_data(*data,idx)
        self.label = category_label(*data,idx)

      def __len__(self):
            return len(self.label)

      def __getitem__(self, index):
            X = self.data[index]
            y = self.label[index]
            return X, y

