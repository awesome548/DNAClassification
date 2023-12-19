import os
import glob
import torch
from torch.utils.data import DataLoader
from ML_dataset.dataset import MultiDataset
from ML_dataset.datamodule import DataModule
from ML_preparation import Preprocess,calu_size
from dotenv import load_dotenv
import pprint
load_dotenv()
DATASIZE = int(os.environ["DATASETSIZE"])
FAST5 = os.environ["FAST5"]
CUTOFF = int(os.environ["CUTOFF"])
MAXLEN = int(os.environ["MAXLEN"])
CUTLEN = int(os.environ["CUTLEN"])
STRIDE = int(os.environ["STRIDE"])
## 変更不可 .values()の取り出しあり
CUTSIZE = {
    "cutoff": CUTOFF,
    "cutlen": CUTLEN,
    "maxlen": MAXLEN,
    "stride": STRIDE,
}

def base_class(fast5_id:list) -> dict:
    d_list = []
    for [fast5, out, flag] in fast5_id:
        pre = Preprocess(fast5,out,flag)
        d_list.append(pre.process())

    dataset_size = calu_size()
    assert d_list[0].shape[0] == dataset_size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]

    train = []
    val = []
    test = []
    for data in d_list:
        tr,v,te = torch.split(data,data_size)
        train.append(tr)
        val.append(v)
        test.append(te)

    return train,val,test,dataset_size

def multi_class(fast5_id:list,ctgys:list) -> dict:
    # データリスト
    d_list = []
    # ラベルリスト
    # seabornに保存する用の変数 [*,*,*,*...]
    l_list = []
    i = 0
    for [fast5, out, flag] in fast5_id:
        for ctgy in ctgys:
            if ctgy in fast5:
                # print(fast5)
                # print(ctgy)
                pre = Preprocess(fast5,out,flag)
                d_list.append(pre.process())
                l_list.append(i)
                i += 1
                break
        l_list.append(-1)

    dataset_size = calu_size()
    assert d_list[0].shape[0] == dataset_size
    data_size = [int(dataset_size*0.8),int(dataset_size*0.1),int(dataset_size*0.1)]

    train = []
    val = []
    test = []
    for data in d_list:
        tr,v,te = torch.split(data,data_size)
        train.append(tr)
        val.append(v)
        test.append(te)

    return train,val,test,dataset_size, l_list

class Dataformat:
    def __init__(self,cls_type:str,use_category:list=False) -> None:
        fast5_set = []
        pprint.pprint(CUTSIZE, width=1)
        """
        fast5_set : [fast5 dir path, species name , torch data exist flag]
        """
        ## 種はターゲットディレクトリに種の名前のフォルダとfast5フォルダを作る
        if os.path.exists(FAST5):
            # Directory starting with A-Z -> loaded : with "_" -> not loaded
            for name in glob.glob(FAST5+'/[A-Z]*'):
                tmp = []
                flag = False
                dirname = os.path.abspath(name) + '/fast5'
                ## IDを作って読み込んでいる場合
                #if os.path.exists(dirname):
                    #flag = True
                    #tmp.append(dirname)
                ### Torchファイルが直接存在する場合<-初回
                #else:
                    #tmp.append(name)
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
        ## 多値分類時との場合わけ
        assert use_category != []
        if use_category:
            train, val, test, dataset_size, l_list = multi_class(fast5_set,use_category)
            cls_type = "multi_value"
        else:
            train, val, test, dataset_size = base_class(fast5_set)
            l_list = None

        self.training_set = MultiDataset(train,cls_type)
        self.validation_set = MultiDataset(val,cls_type)
        self.test_set = MultiDataset(test,cls_type)
        ## カテゴリの数がDatasetのclass属性に保存してある
        self.classes = MultiDataset.classes
        ## seabornに保存する用の変数 [*,*,*,*...]
        if l_list:
            captions = l_list
        else:
            captions = MultiDataset.captions

        y_label = [None]*(self.classes)
        for spe,cap in zip(fast5_set,captions):
            if not(cap < 0):
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


