from process import Preprocess,calu_size
import torch
import glob
from dataset.dataset import MultiDataset
from dataset.datamodule import DataModule
from torch.utils.data import DataLoader

def two_class(idset: list,dataset:list,size:int,cut_size:dict) -> dict:
    assert len(dataset) == 2

    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataset_size = calu_size(**cut_size,size=size)*size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    # g_cpu = torch.Generator().manual_seed(1234567)
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)

    train = [train_A,train_B]
    val = [val_A,val_B]
    test = [test_A,test_B]
    
    return train,val,test,dataset_size

def two_class_concat(idset:list,dataset:list,size:int,cut_size:dict) -> dict:
    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataC = Preprocess(idset[2]).process(inpath=dataset[2],**cut_size,size=size)
    dataD = Preprocess(idset[3]).process(inpath=dataset[3],**cut_size,size=size)

    dataB = torch.vstack([dataB,dataC,dataD])
    dataset_size = calu_size(**cut_size,size=size)*size
    
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    train_A, val_A, test_A = torch.split(dataA,data_size)
    data_size = [int(dataset_size*0.8*3),int(dataset_size*0.1*3),int(dataset_size*0.1*3)]
    train_B, val_B, test_B = torch.split(dataB,data_size)

    train = [train_A,train_B]
    val = [val_A,val_B]
    test = [test_A,test_B]
    
    return train,val,test,dataset_size

def three_class(idset: list,dataset:list,size:int,cut_size:dict) -> dict:
    assert len(dataset) >= 3
    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataC = Preprocess(idset[2]).process(inpath=dataset[2],**cut_size,size=size)
    dataset_size = calu_size(**cut_size,size=size)*size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    # g_cpu = torch.Generator().manual_seed(1234567)
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)
    train_C, val_C, test_C = torch.split(dataC,data_size)

    train = [train_A,train_B,train_C]
    val = [val_A,val_B,val_C]
    test = [test_A,test_B,test_C]
    
    return train,val,test,dataset_size


def four_class(idset: list,dataset:list,size:int,cut_size:dict) -> dict:
    assert len(dataset) == 4
    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataC = Preprocess(idset[2]).process(inpath=dataset[2],**cut_size,size=size)
    dataD = Preprocess(idset[3]).process(inpath=dataset[3],**cut_size,size=size)
    dataset_size = calu_size(**cut_size, size=size)*size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)
    train_C, val_C, test_C = torch.split(dataC,data_size)
    train_D, val_D, test_D = torch.split(dataD,data_size)

    train = [train_A,train_B,train_C,train_D]
    val = [val_A,val_B,val_C,val_D]
    test = [test_A,test_B,test_C,test_D]
    
    return train,val,test,dataset_size


def base_class(idset: list,dataset:list,size:int,cut_size:dict,base:int) -> dict:
    assert len(dataset) >= base
    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataC = Preprocess(idset[2]).process(inpath=dataset[2],**cut_size,size=size)
    dataD = Preprocess(idset[3]).process(inpath=dataset[3],**cut_size,size=size)
    dataE = Preprocess(idset[4]).process(inpath=dataset[4],**cut_size,size=size)
    dataF = Preprocess(idset[5]).process(inpath=dataset[5],**cut_size,size=size)
    dataset_size = calu_size(**cut_size, size=size)*size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)
    train_C, val_C, test_C = torch.split(dataC,data_size)
    train_D, val_D, test_D = torch.split(dataD,data_size)
    train_E, val_E, test_E = torch.split(dataE,data_size)
    train_F, val_F, test_F = torch.split(dataF,data_size)

    assert train_A.shape == train_B.shape == train_C.shape == train_D.shape == train_E.shape == train_F.shape

    train = [train_A,train_B,train_C,train_D,train_E,train_F]
    val = [val_A,val_B,val_C,val_D,val_E,val_F]
    test = [test_A,test_B,test_C,test_D,test_E,test_F]
    
    return train,val,test,dataset_size


class Dataformat:
    def __init__(self,target: list,inpath:list,dataset_size:int,cut_size:dict,num_classes:int,base_classes:int) -> None:
        idset = glob.glob(target+'/*.txt')
        dataset = glob.glob(inpath+'/*')
        idset.sort()
        dataset.sort()

        train, val, test, dataset_size = base_class(idset,dataset,dataset_size,cut_size,base_classes)

        self.training_set = MultiDataset(train,num_classes,base_classes)
        self.validation_set = MultiDataset(val,num_classes,base_classes)
        self.test_set = MultiDataset(test,num_classes,base_classes)
        #val_dataset_size = len(self.training_set)+len(self.validation_set)+len(self.test_set)
        #assert val_dataset_size == dataset_size*base_classes
        self.dataset = dataset_size
        pass

    def module(self,batch):
        return DataModule(self.training_set,self.validation_set,self.test_set,batch_size=batch)

    def loader(self,batch):
        params = {'batch_size': batch,
				'shuffle': True,
				'num_workers': 24}
        return DataLoader(self.training_set,**params),DataLoader(self.validation_set,**params),DataLoader(self.test_set,**params)

    def size(self) -> int:
        return self.dataset


