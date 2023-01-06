import torch
import torch.nn as nn
import torch.utils.data
import optuna 
import ray
from ray import tune, air
from ray.air import session
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from models import resnet,effnetv2_s
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import setup_wandb
import wandb
from optim.utils import resnet_param,effnet_param,resnet_var,effnet_var

def train_loop(model, device, train_loader, criterion,optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        wandb.log({"train_loss": loss})
        loss.backward()
        optimizer.step()
    return loss

def test_loop(model, device, test_loader,criterion,target_class):
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
            wandb.log("test_loss",loss)
            y_hat_idx = output.max(dim=1).indices
            y_hat_idx = (y_hat_idx == target_class)
            y = (target == target_class)
            tp += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
            fp += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx != y))
            tn += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx == y))
            fn += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))
        wandb.log("test_Accuracy",(tp+tn)/(tp+tn+fp+fn))
        wandb.log("test_Recall",(tp)/(tp+fn))
        wandb.log("test_Precision",(tp)/(tp+fp))
        wandb.log("test_F1",2*( (tp)/(tp+fp) * (tp)/(tp+fn) ) / ( (tp)/(tp+fp) + (tp)/(tp+fn) ))

def objective(config:dict):
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idpath = "/z/kiku/Dataset/ID"
    inpath ="/z/kiku/Dataset/Target"
    arch = "Effnet"
    batch = 100
    cutoff = 1500
    classes = 2
    target_class = 1
    heatmap = False
    cutoff = 1500
    EPOCH = 30
    preference = {
        "lr" : lr,
        "cutlen" : cutlen,
        "classes" : classes,
        "epoch" : EPOCH,
        "target" : target_class,
        "name" : arch,
        "heatmap" : heatmap,
    }
    #cutlen,conv_1,conv_2,conv_3,conv_4,layer_1,layer_2,layer_3,layer_4,learningrate,cfgs = resnet_var(config)
    #model = resnet(preference,cfgs)
    cutlen,cnnparams,mode,lr = effnet_var(config)
    model = effnetv2_s(mode,preference,cnnparams)

    dataset_size,cut_size = data_preference(cutoff,cutlen)
    data = Dataformat(idpath,inpath,dataset_size,cut_size,num_classes=classes,base_classes=4)
    train_loader,_,test_loader = data.loader(batch)
    dataset_size = data.size()
    
    # network, loss functions and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCH):
        print('Epoch {}'.format(epoch+1))
        # training
        train_loss = train_loop(model, device, train_loader, criterion,optimizer)
        # testing with validation data
        test_loop(model, device, test_loader,criterion,target_class)
        # report
        session.report({"loss":train_loss})

    return model, optimizer

def main():
    wandb.init(project="RayTune-optim")
    #search_space = resnet_param()
    search_space = effnet_param()
    #trainable_with_resources = tune.with_resources(objective, {"GPU": 1})
    ray.init(num_gpus=1,num_cpus=24)
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(space=search_space,metric="loss", mode="min"),
            num_samples=1,
            scheduler=ASHAScheduler(
                metric="loss", mode="min",max_t=40,
                grace_period=10
            ),
        ),
        run_config=air.RunConfig(
            stop={"training_iteration":20},
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

if __name__ == '__main__':
    main()