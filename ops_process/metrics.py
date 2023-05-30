from torchmetrics.classification import MulticlassConfusionMatrix,MulticlassAccuracy,MulticlassPrecision,MultilabelPrecision
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.cluster import KMeans
import datetime
from dotenv import load_dotenv

load_dotenv()
CONFMAT = os.environ['CONFMAT']
HEATMAP = os.environ['HEATMAP']

def evaluation(y_hat_idx,y_hat,y,n_class,target,hidd_vec,labels,pref,load_model,writer):
    acc = MulticlassAccuracy(num_classes=n_class)
    acc1 = MulticlassAccuracy(num_classes=n_class,average=None)
    preci = MulticlassPrecision(num_classes=n_class)
    preci1 = MulticlassPrecision(num_classes=n_class,average=None)
    # recall = Recall(task="multiclass",num_classes=n_class)
    # recall1 = Recall(task="multiclass",num_classes=n_class,average=None)
    # f1 = F1Score(task="multiclass",num_classes=n_class,average=None)
    # auroc = AUROC(task="multiclass", num_classes=n_class)
    # auroc1 = AUROC(task="multiclass", num_classes=n_class,average=None)
    print(f'Metric_AccuracyMacro : {acc(y_hat_idx,y)}')
    print(f'Metric_AccuracyMicro : {acc1(y_hat_idx,y)[target]}')
    print(f'Metric_PrecisionMacro : {preci(y_hat_idx,y)}')
    print(f'Metric_PrecisionMicro : {preci1(y_hat_idx,y)[target]}')
    # print(f'Metric_RecallMacro : {recall(y_hat_idx,y)}')
    # print(f' Metric_RecallMicro : {recall1(y_hat_idx,y)[target]}')
    # print(f'Metric_F1 : {f1(y_hat_idx,y)[target]}')
    # print(f'Metric_AurocMacro : {auroc(y_hat,y)}')
    # print(f'Metric_AurocMicro : {auroc1(y_hat,y)[target]}')
    writer.add_scalar("Metric/accuracy",acc(y_hat_idx,y))
    writer.add_scalar("Metric/accuracy_tar",acc1(y_hat_idx,y)[target])
    writer.add_scalar("Metric/precision",preci(y_hat_idx,y))
    writer.add_scalar("Metric/precision_tar",preci1(y_hat_idx,y)[target])
            
           
    _,cutlen,n_class,epoch,target,name,heatmap,project = pref.values()
    confmat_norm = MulticlassConfusionMatrix(num_classes=n_class,normalize='true')
    matrix = confmat_norm(y_hat_idx, y)
    matrix = matrix.cpu().detach().numpy().copy()
    print(matrix)
    os.makedirs(f"{CONFMAT}/{project}",exist_ok=True)
    plt.figure()
    s = sns.heatmap(matrix,annot=True,cmap="Reds",fmt=".2f")
    s.set(xlabel="predicted",ylabel="label")
    plt.savefig(f"{CONFMAT}/{project}/{datetime.date.today()}{name}-{str(cutlen)}-e{epoch}-c{n_class}.png")


    ### K-Means ###
    if heatmap and (not load_model):
        print("saving heatmap...")
        X = hidd_vec.cpu().detach().numpy().copy()
        heat_map = torch.zeros(n_class,n_class)

        kmeans = KMeans(n_clusters=n_class,init='k-means++',n_init=1,random_state=0).fit(X)
        val_len = 0
        for i in range(n_class):
            p = labels[kmeans.labels_ ==i]
            val_len += int(p.shape[0])
            for j in range(n_class):
                x = torch.zeros(p.shape)
                x[p==j] = 1
                heat_map[i,j] = torch.count_nonzero(x)
        assert val_len == int(labels.shape[0])

        for i in range(n_class):
            heat_map[:,i] = heat_map[:,i]/heat_map.sum(0)[i]
        heatmap = heat_map.cpu().detach().numpy().copy()

        os.makedirs(f"{HEATMAP}/{project}",exist_ok=True)
        ### SAVE FIG ###
        plt.figure()
        s = sns.heatmap(heatmap,vmin=0.0,vmax=1.0,annot=True,cmap="Reds",fmt=".3g")
        s.set(xlabel="label",ylabel="cluster")
        plt.savefig(f"{HEATMAP}/{project}/{datetime.date.today()}{name}-{str(cutlen)}-e{epoch}-c{n_class}.png")
        print("heatmap saved...")
        ### SAVE FIG ###
    else:
        print("heatmap will not be saved")
