import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pprint
import click
import csv
import numpy as np
import torch
import glob
from dotenv import load_dotenv
from torch import nn
from ML_processing import test_loop, train_loop
from utils import data_prep,preference_prep,text_writer

@click.command()
## --- frequently change
@click.option("--arch", "-a",default="ResNet" ,help="Name of Architecture")
@click.option("--mode", "-m", default=1, help="0,1,2,3 : CNN kernel order,family : dataset categorization")
@click.option("--reps", "-r", default=10, help="training reps")
## --- rarely change
@click.option("--batch", "-b", default=1000, help="Batch size, default 1000")
@click.option("--minepoch", "-e", default=20, help="Number of min epoches")
@click.option("--learningrate", "-lr", default=1e-2, help="Learning rate")
@click.option("--hidden", "-hidden", default=64, help="dim of hidden layer")
@click.option("--t_class", "-t", default=0, help="Target class index")
@click.option("--cls_type", "-c", default="base", help="base, genus, family")
@click.option("--project", "-p", default="misc", help="")
@click.option("--layers", "-l", default=4, help="")

def main(arch, batch, minepoch, learningrate, hidden, t_class, mode, cls_type,reps,project,layers):
    # print(torch.cuda.device_count())
    load_dotenv()
    cutlen = int(os.environ["CUTLEN"])
    FAST5 = os.environ["FAST5"]
    RESULT = os.environ["RESULT"]
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

    if not os.path.isdir(f'{RESULT}/{project}'):
        os.makedirs(f'{RESULT}/{project}')
    ## ファイルの順番がわからなくなるためソート
    fast5_set.sort()
    # print(fast5_set)
    result = np.zeros((len(fast5_set)-1,len(fast5_set)))

    for idx, category1 in enumerate(fast5_set):
        if idx == len(fast5_set):
            break
        for idx2 in range(idx+1,len(fast5_set)):
            category2 = fast5_set[idx2]
            use_category = (category1,category2)

            tmp_result = []
            for rep in range(int(reps)):
                print(use_category)
                train_loader, val_loader,test_loader, param = data_prep(cls_type,use_category,batch)
                model, pref = preference_prep(arch,hidden,mode,learningrate,minepoch,t_class,cls_type,project,layers,param)

                if torch.cuda.is_available():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                models = {
                    "model": model.to(device),
                    "criterion": nn.CrossEntropyLoss().to(device),
                    "optimizer": torch.optim.Adam(model.parameters(), lr=learningrate),
                    "device": device,
                }

                _ = train_loop(models, pref, train_loader, val_loader, load_model)
                acc = test_loop(models, pref, test_loader, load_model, use_category)
                tmp_result.append(acc)
                filename = f'{RESULT}/{project}/{category1}_{category2}.txt'
                text_writer(filename,acc)

            assert reps == len(tmp_result) 
            result[idx][idx2] = sum(tmp_result)/int(reps)

    print(result)
    headers = [""] + fast5_set
    data_with_headers = [headers]

    result = result.tolist()
    for i, row in enumerate(result):
        data_with_headers.append([fast5_set[i]] + row)
    with open(f'{RESULT}/{pref["project"]}/result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_with_headers)

if __name__ == "__main__":
    main()
