import torch
import torch.nn  as nn
import time
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from ML_model import LSTM,resnet,SimpleViT,ViT,ViT2,SimpleViT2,Transformer_clf_model,GRU
from optuna.trial import TrialState
import torch.utils.data
import optuna 
from dataset.dataformat import Dataformat
from ML_model import resnet,effnetv2_s
import wandb
from optim.utils import resnet_param,effnet_param


### TRAIN and TEST ###
def train_loop(model, device, train_loader, criterion,optimizer,run):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        run.log({"train_loss": loss})
    return loss

def test_loop(model, device, test_loader,criterion,target_class,run):
    model.eval()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output,target)
            run.log({"test_loss":loss})
            y_hat_idx = output.max(dim=1).indices
            y_hat_idx = (y_hat_idx == target_class)
            y = (target == target_class)
            tp += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
            fp += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx != y))
            tn += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx == y))
            fn += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))
        
        r = (tp)/(tp+fn)
        p = (tp)/(tp+fp)
        f1 = 2*(r*p)/(r+p)
        run.log({
            "test_Accuracy":(tp+tn)/(tp+tn+fp+fn),
            "test_Recall": r,
            "test_Precision": p,
            "test_F1": f1,
        })

    return f1

### varible ###
IDPATH = "/z/kiku/Dataset/ID"
INPATH ="/z/kiku/Dataset/Target"
ARCH = "Effnet"
BATCH = 100
EPOCH = 20
CUTOFF = 1500
CLASSES = 2
TARGET = 0
HEATMAP = False
PROJECT = "Category45-effnet-optim"

def objective(trial):
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cutlen,mode,lr,cnn_params = effnet_param(trial)
    #cutlen,mode,lr,cnn_params,cfgs = resnet_param(trial)
    run = wandb.init(
        project=PROJECT,
        config={
            "cutlen" : cutlen,
            "mode" : mode,
            "lr" : lr,
            "cnnparam" : cnn_params,
            #"cfgs" : cfgs
        },
        reinit=True,
    )
    with run:
        print(f"cutlen : {cutlen}")
        print(f"mode : {mode}")
        print(f"lr : {lr}")
        print(f"cnn_param : {cnn_params}")
        #print(f"cfgs : {cfgs}")
        preference = {
            "lr" : lr,
            "cutlen" : cutlen,
            "classes" : CLASSES,
            "epoch" : EPOCH,
            "target" : TARGET,
            "name" : ARCH,
            "heatmap" : HEATMAP,
        }
        #model = resnet(preference,cnnparam=cnn_params,mode=mode,cfgs=cfgs)
        model = effnetv2_s(mode[0],preference,cnn_params)

        dataset_size,cut_size = data_preference(CUTOFF,cutlen)
        data = Dataformat(IDPATH,INPATH,dataset_size,cut_size,num_classes=CLASSES)
        train_loader,_,test_loader = data.loader(BATCH)
        dataset_size = data.size()
    
        # network, loss functions and optimizer
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print("Train Start...")
        train_loss = 0
        for epoch in range(EPOCH):
            print('Epoch {}'.format(epoch+1))
            train_loss = train_loop(model, device, train_loader, criterion,optimizer,run)

        # testing with validation data
        print("Test Start...")
        f1 = test_loop(model, device, test_loader,criterion,TARGET,run)
        # report
    return f1

if __name__ == "__main__":
    time_sta = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    time_end = time.time()

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))