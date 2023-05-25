from torchmetrics import Accuracy,Recall,Precision,F1Score,ConfusionMatrix,AUROC
import wandb
import torch
from sklearn.cluster import KMeans

def evaluation(y_hat_idx,y_hat,y,n_class,target,cluster,labels):
        acc = Accuracy(task="multiclass",num_classes=n_class)
        acc1 = Accuracy(task="multiclass",num_classes=n_class,average=None)
        preci = Precision(task="multiclass",num_classes=n_class)
        preci1 = Precision(task="multiclass",num_classes=n_class,average=None)
        recall = Recall(task="multiclass",num_classes=n_class)
        recall1 = Recall(task="multiclass",num_classes=n_class,average=None)
        f1 = F1Score(task="multiclass",num_classes=n_class,average=None)
        auroc = AUROC(task="multiclass", num_classes=n_class)
        auroc1 = AUROC(task="multiclass", num_classes=n_class,average=None)
        confmat = ConfusionMatrix(task="multiclass",num_classes=n_class)
        confmat = confmat(y_hat_idx, y)


        ### K-Means ###
        if heatmap:
            cluster 
            labels 
            X = cluster.cpu().detach().numpy().copy()
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
            plt.savefig(f"{HEATMAP}/{project}/{name}-{str(cutlen)}-e{epoch}-c{n_class}-{inference_time}.png")
            ### SAVE FIG ###
        confmat = confmat.cpu().detach().numpy().copy()
        os.makedirs(f"{CONFMAT}/{project}",exist_ok=True)
        plt.figure()
        s = sns.heatmap(confmat,annot=True,cmap="Reds",fmt="d")
        s.set(xlabel="predicted",ylabel="label")
        plt.savefig(f"{CONFMAT}/{project}/{name}-{str(cutlen)}-e{epoch}-c{n_class}-{inference_time}.png")
