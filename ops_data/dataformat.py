import os
import glob
import torch
from torch.utils.data import DataLoader
from ops_data.dataset import MultiDataset
from ops_data.datamodule import DataModule
from ops_data.preprocess import Preprocess,calu_size

def base_class(ids: list,fast5s:list,dataset_size:int,cut_size:dict) -> dict:
    cutoff,cutlen,maxlen,stride = cut_size.values()
    data_list = []
    for id, fast5 in zip(ids,fast5s):
        pre = Preprocess(id,fast5)
        if len(pre) >= dataset_size:
            data_list.append(pre.process(**cut_size,req_size=dataset_size))
        else:
            raise IndexError('dataset size is larger than actual num of fast5')
    manipulate_ratio = calu_size(cutlen,maxlen,stride)
    dataset_size = manipulate_ratio*dataset_size

    assert data_list[0].shape[0] == dataset_size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]
    
    train = []
    val = []
    test = []
    for data in data_list:
        tr,v,te = torch.split(data,data_size)
        train.append(tr)
        val.append(v)
        test.append(te)
    
    return train,val,test,dataset_size

class Dataformat:
    def __init__(self,ids_dir: list,fast5_dir:list,dataset_size:int,cut_size:dict,type:str) -> None:
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

        #ファイルの順番がわからなくなるためソート
        fast5_set.sort()
        id_set.sort()
        train, val, test, dataset_size = base_class(id_set,fast5_set,dataset_size,cut_size)

        #カテゴリの数がDatasetのclass属性に保存してある
        self.training_set = MultiDataset(train,type)
        self.validation_set = MultiDataset(val,type)
        self.test_set = MultiDataset(test,type)
        self.classes = self.training_set.classes

        ### DATASET SIZE VALIDATON 
        val_datsize = len(self.training_set)+len(self.validation_set)+len(self.test_set)
        num_class = len(id_set)
        tmp_size = dataset_size *num_class
        assert val_datsize == tmp_size

        self.dataset = dataset_size
        pass

    def module(self,batch):
        return DataModule(self.training_set,self.validation_set,self.test_set,batch_size=batch)

    def loader(self,batch):
        params = {'batch_size': batch,
				'shuffle': True,
				'num_workers': 24}
        return DataLoader(self.training_set,**params),DataLoader(self.validation_set,**params)
    
    def test_loader(self,batch):
        params = {'batch_size': batch,
				'shuffle': False,
				'num_workers': 24}
        return DataLoader(self.test_set,**params)

    def size(self) -> int:
        return self.dataset

    def __len__(self) -> int:
        return self.classes


