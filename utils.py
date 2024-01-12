import torch
from ML_dataset import Dataformat
from ML_model import model_preference
from dotenv import load_dotenv
import pprint
import os
load_dotenv()
CUTLEN = int(os.environ["CUTLEN"])

def load_model(model_path, model_class):
    # モデルのインスタンスを作成
    model = model_class()
    # モデルの状態をロード
    model.load_state_dict(torch.load(model_path))
    # 推論モードに設定
    model.eval()
    return model

def data_prep(cls_type,use_category,batch):
    print("##### DATA PREPARATION #####")
    data = Dataformat(cls_type,use_category)
    train_loader, val_loader = data.loader(batch)
    test_loader = data.test_loader(batch)
    param = data.param()
    print(f'Num of Classes :{param["num_cls"]}')
    return train_loader,val_loader, test_loader,param

def preference_prep(arch,hidden,mode,lr,epo,t_cls,cls_type,project,lay,param):
    print("##### PREFERENCE #####")

    pref = {
        "data_size": param['size'],
        "lr": lr,
        "cutlen": CUTLEN,
        "classes": param['num_cls'],
        "epoch": epo,
        "target": t_cls,
        "confmat": False,
        "heatmap": False,
        "y_label": param['ylabel'],
        "project": project,
        "category" : cls_type,
        "layers" : lay,
    }
    model, useModel = model_preference(arch, hidden, pref, mode=mode)
    pref['name'] = useModel
    pprint.pprint(pref, width=1)

    return model,pref

def text_writer(filename,data,option=None,**kwgs):
    with open(filename, 'a') as f:
        if option != None:
            f.write(f"--{option}--\n")
        f.write(f"{data}\n")