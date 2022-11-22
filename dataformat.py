from preprocess import Preprocess
import torch
from dataset import MultiDataset
from datamodule import DataModule
from preprocess import calu_size
def two_class(idset: list,dataset:list,size:int,cut_size:dict) -> dict:
    assert len(dataset) == 2

    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataset_size = calu_size(size,**cut_size)*size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    # g_cpu = torch.Generator().manual_seed(1234567)
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)

    train = {
        "a" : train_A,
        "b" : train_B,
    }
    val = {
        "a" : val_A,
        "b" : val_B,
    }
    test = {
        "a" : test_A,
        "b" : test_B,
    }
    return train,val,test,dataset_size

def three_class(idset: list,dataset:list,size:int,cut_size:dict) -> dict:
    assert len(dataset) == 3
    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataC = Preprocess(idset[2]).process(inpath=dataset[2],**cut_size,size=size)
    dataset_size = calu_size(**cut_size,size=size)*size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    # g_cpu = torch.Generator().manual_seed(1234567)
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)
    train_C, val_C, test_C = torch.split(dataC,data_size)

    train = {
        "a" : train_A,
        "b" : train_B,
        "c" : train_C,
    }
    val = {
        "a" : val_A,
        "b" : val_B,
        "c" : val_C,
    }
    test = {
        "a" : test_A,
        "b" : test_B,
        "c" : test_C,
    }
    return train,val,test,dataset_size


def four_class(idset: list,dataset:list,size:int,cut_size:dict) -> dict:
    assert len(dataset) == 4
    dataA = Preprocess(idset[0]).process(inpath=dataset[0],**cut_size,size=size)
    dataB = Preprocess(idset[1]).process(inpath=dataset[1],**cut_size,size=size)
    dataC = Preprocess(idset[2]).process(inpath=dataset[2],**cut_size,size=size)
    dataD = Preprocess(idset[3]).process(inpath=dataset[3],**cut_size,size=size)
    dataset_size = calu_size(size,**cut_size)*size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    # g_cpu = torch.Generator().manual_seed(1234567)
    train_A, val_A, test_A = torch.split(dataA,data_size)
    train_B, val_B, test_B = torch.split(dataB,data_size)
    train_C, val_C, test_C = torch.split(dataC,data_size)
    train_D, val_D, test_D = torch.split(dataD,data_size)

    train = {
        "a" : train_A,
        "b" : train_B,
        "c" : train_C,
        "d" : train_D,
    }
    val = {
        "a" : val_A,
        "b" : val_B,
        "c" : val_C,
        "d" : val_D,
    }
    test = {
        "a" : test_A,
        "b" : test_B,
        "c" : test_C,
        "d" : test_D,
    }
    return train,val,test,dataset_size

class Dataformat:
    def __init__(self,idset: list,dataset:list,dataset_size:int,cut_size:dict,num_classes:int,transform:dict) -> None:
        idset.sort()
        dataset.sort() 

        if num_classes == 2:
            train, val, test,dataset_size = two_class(idset,dataset,dataset_size,cut_size)
        elif num_classes ==3:
            train, val ,test,dataset_size = three_class(idset,dataset,dataset_size,cut_size)
        elif num_classes ==4:
            train, val,test,dataset_size = four_class(idset,dataset,dataset_size,cut_size)


        self.training_set = MultiDataset(train,num_classes,transform)
        self.validation_set = MultiDataset(val,num_classes,transform)
        self.test_set = MultiDataset(test,num_classes,transform)
        self.dataset = dataset_size
        pass
    

    def process(self,batch):
        return DataModule(self.training_set,self.validation_set,self.test_set,batch_size=batch)
    
    def size(self) -> int:
        return self.dataset

    