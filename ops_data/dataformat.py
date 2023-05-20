import os
import glob
import torch
from torch.utils.data import DataLoader
from ops_data.dataset import MultiDataset
from ops_data.datamodule import DataModule
from ops_process import Preprocess,calu_size

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

class Dataformat:
    def __init__(self,ids_dir: list,fast5_dir:list,dataset_size:int,cut_size:dict,num_classes:int) -> None:
        fast5_set = []
        id_set = []
        if os.path.exists(fast5_dir):
            for name in glob.glob(fast5_dir+'/*'):
                dirname = os.path.abspath(name) + '/fast5'
                if os.path.exists(dirname):
                    fast5_set.append(dirname)
                    id_set.append(ids_dir + f'/{os.path.basename(name)}.txt')
        else:
            raise FileNotFoundError("ディレクトリがありません")

        id_set.sort()
        fast5_set.sort()

        train, val, test, dataset_size = base_class(id_set,fast5_set,dataset_size,cut_size)

        self.training_set = MultiDataset(train,num_classes)
        self.validation_set = MultiDataset(val,num_classes)
        self.test_set = MultiDataset(test,num_classes)
        val_datsize = len(self.training_set)+len(self.validation_set)+len(self.test_set)
        tmp_size = (dataset_size * 7) if num_classes >= 5 else (dataset_size *num_classes)
        assert val_datsize == tmp_size
        self.dataset = dataset_size
        pass

    def module(self,batch):
        return DataModule(self.training_set,self.validation_set,self.test_set,batch_size=batch)

    def loader(self,batch):
        params = {'batch_size': batch,
				'shuffle': True,
				'num_workers': 24}
        return DataLoader(self.training_set,**params),DataLoader(self.validation_set,**params),DataLoader(self.test_set,**params)
    
    def test_loader(self,batch):
        params = {'batch_size': batch,
				'shuffle': False,
				'num_workers': 24}
        return DataLoader(self.test_set,**params)

    def size(self) -> int:
        return self.dataset


