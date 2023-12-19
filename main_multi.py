import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import pprint
import click
import numpy as np
import torch
import glob
import datetime
from dotenv import load_dotenv
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ML_dataset import Dataformat
from ML_processing import test_loop, train_loop
from ML_model import model_preference

@click.command()
## --- frequently change
@click.option("--arch", "-a",default="ResNet" ,help="Name of Architecture")
@click.option("--mode", "-m", default="normal", help="0,1,2,3 : CNN kernel order,family : dataset categorization")
@click.option("--reps", "-r", default=5, help="training reps")

## --- rarely change
@click.option("--batch", "-b", default=1000, help="Batch size, default 1000")
@click.option("--minepoch", "-e", default=20, help="Number of min epoches")
@click.option("--learningrate", "-lr", default=1e-2, help="Learning rate")
@click.option("--hidden", "-hidden", default=64, help="dim of hidden layer")
@click.option("--t_class", "-t", default=0, help="Target class index")
@click.option("--cls_type", "-c", default="base", help="base, genus, family")
@click.option("--project", "-p", default="base", help="base, genus, family")
@click.option("--ctgy1", "-c1", default="n", help="")
@click.option("--ctgy2", "-c2", default='n', help="")
@click.option("--ctgy3", "-c3", default='n', help="")
@click.option("--ctgy4", "-c4", default='n', help="")
@click.option("--layers", "-l", default=1, help="")


def main(arch, batch, minepoch, learningrate, hidden, t_class, mode, cls_type,reps,project,ctgy1,ctgy2,ctgy3,ctgy4,layers):
    load_dotenv()
    cutlen = int(os.environ["CUTLEN"])
    load_model = False
    

    # print(torch.cuda.device_count())
    load_dotenv()
    cutlen = int(os.environ["CUTLEN"])
    FAST5 = os.environ["FAST5"]
    load_model = False
    
    fast5_set = []
    """
    fast5_set : [fast5 dir path, species name , torch data exist flag]
    """
    ## 種はターゲットディレクトリに種の名前のフォルダとfast5フォルダを作る
    if os.path.exists(FAST5):
        # Directory starting with A-Z -> loaded : with "_" -> not loaded
        for name in glob.glob(FAST5+'/[A-Z]*'):
            fast5_set.append(os.path.basename(name))
    else:
        raise FileNotFoundError("ディレクトリがありません")

    ## ファイルの順番がわからなくなるためソート
    fast5_set.sort()

    use_category = [ctgy1,ctgy2,ctgy3,ctgy4]
    use_category = [x for x in use_category if x != "n"]

    for lay in range(1,int(layers)+1):
        if not os.path.isdir(f'result/{project}_{lay}'):
            os.makedirs(f'result/{project}_{lay}')
        filename = f'result/{project}_{lay}/{use_category[0]}_{lay}_{datetime.date.today()}.txt'
        tmp_result = []
        for rep in range(int(reps)):
            #print(use_category)
            """
            Dataset preparation
            データセット設定
            """
            ## fast5 -> 種のフォルダが入っているディレクトリ -> 対応の種のみを入れたディレクトリを使うこと！！
            ## id list -> 種の名前に対応した.txtが入ったディレクトリ
            print("##### DATA PREPARATION #####")
            data = Dataformat(cls_type,use_category)
            train_loader, val_loader = data.loader(batch)
            test_loader = data.test_loader(batch)
            param = data.param()
            datasize, classes, ylabel = param["size"], param["num_cls"], param["ylabel"]
            print(f"Num of Classes :{classes}")
            """
            Preference
            Model設定
            """
            ## 変更不可 .values()の取り出しあり evaluation.py
            print("##### PREFERENCE #####")
            pref = {
                "data_size": datasize,
                "lr": learningrate,
                "cutlen": cutlen,
                "classes": classes,
                "epoch": minepoch,
                "target": t_class,
                "name": arch,
                "confmat": False,
                "heatmap": False,
                "y_label": ylabel,
                "project": project,
                "category" : cls_type,
                "layers" : lay,
            }
            pprint.pprint(pref, width=1)
            model, useModel = model_preference(arch, hidden, pref, mode=mode)
            """
            Training
            """
            ### Train ###
            if torch.cuda.is_available:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            ### network, loss functions and optimizer
            ## 変更不可 .values()の取り出しあり loop.py
            models = {
                "model": model.to(device),
                "criterion": nn.CrossEntropyLoss().to(device),
                "optimizer": torch.optim.Adam(model.parameters(), lr=learningrate),
                "device": device,
            }
            train_loop(models, pref, train_loader, val_loader, load_model)
            acc = test_loop(models, pref, test_loader, load_model, use_category)
            tmp_result.append(acc)

            # テキストファイルにaccuracyを書き込む
            with open(filename, 'a') as f:
                f.write(f"{acc}\n")

        result = sum(tmp_result)/int(reps)
        with open(filename, 'a') as f:
            f.write("--FINAL RESULT--\n")
            f.write(f"{result}\n")





if __name__ == "__main__":
    main()