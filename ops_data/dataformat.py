import os
import glob
import torch
from torch.utils.data import DataLoader
from ops_data.dataset import MultiDataset
from ops_data.datamodule import DataModule
from ops_data.preprocess import Preprocess,calu_size

def base_class(fast5_id:list,dataset_size:int,cut_size:dict) -> dict:
    cutoff,cutlen,maxlen,stride = cut_size.values()
    data_list = []
    for [fast5, out, flag] in fast5_id:
        pre = Preprocess(fast5,out,flag)
        data_list.append(pre.process(**cut_size,req_size=dataset_size))
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

def two_class(fast5_id:list,dataset_size:int,cut_size:dict,cat1:str,cat2) -> dict:
    _,cutlen,maxlen,stride = cut_size.values()
    data_list = []
    for [fast5, out, flag] in fast5_id:
        if (cat1 in fast5) or (cat2 in fast5):
            pre = Preprocess(fast5,out,flag)
            data_list.append(pre.process(**cut_size,req_size=dataset_size))
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
    def __init__(self,fast5_dir:list,dataset_size:int,cut_size:dict,classfi_type:str,use_category:str=False) -> None:
        fast5_set = []
        """
        fast5_set : [fast5 dir path, species name , torch data exist flag]
        """
        ## 種はターゲットディレクトリに種の名前のフォルダとfast5フォルダを作る
        if os.path.exists(fast5_dir):
            # Directory starting with A-Z -> loaded : with "_" -> not loaded
            for name in glob.glob(fast5_dir+'/[A-Z]*'):
                tmp = []
                flag = False
                dirname = os.path.abspath(name) + '/fast5'
                ## IDを作って読み込んでいる場合
                if os.path.exists(dirname):
                    flag = True
                    tmp.append(dirname)
                ## Torchファイルが直接存在する場合<-初回
                else:
                    tmp.append(name)
                tmp.append(os.path.basename(name))
                tmp.append(flag)
                fast5_set.append(tmp)
        else:
            raise FileNotFoundError("ディレクトリがありません")

        ## ファイルの順番がわからなくなるためソート
        fast5_set.sort()
        # print(fast5_set)

        ## 二値分類時との場合わけ
        if use_category:
            train, val, test, dataset_size = two_class(fast5_set,dataset_size,cut_size,*use_category)
            classfi_type = "two_value"
        else:
            train, val, test, dataset_size = base_class(fast5_set,dataset_size,cut_size)

        self.training_set = MultiDataset(train,classfi_type)
        self.validation_set = MultiDataset(val,classfi_type)
        self.test_set = MultiDataset(test,classfi_type)
        ## カテゴリの数がDatasetのclass属性に保存してある
        self.classes = MultiDataset.classes
        ## seabornに保存する用の変数 [*,*,*,*...]
        captions = MultiDataset.captions
        y_label = [None]*(self.classes)
        for spe,cap in zip(fast5_set,captions):
            if y_label[cap] == None:
                y_label[cap] = spe[1]
            else:
                y_label[cap] += f'\n{spe[1]}'
        self.ylabel = y_label

        ### DATASET SIZE VALIDATON
        ## 二値分類では使えないためコメントアウト
        # val_datsize = len(self.training_set)+len(self.validation_set)+len(self.test_set)
        # num_class = len(fast5_set)
        # tmp_size = dataset_size *num_class
        # assert val_datsize == tmp_size

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

    def param(self) -> int:
        return {"size":self.dataset,"num_cls":self.classes,"ylabel":self.ylabel}


