import torch
import torch.nn as nn
import torch.utils.data
import optuna 
from ray import tune, air
from ray.air import session
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from models import LSTM,resnet,SimpleViT,ViT,ViT2,SimpleViT2,Transformer_clf_model,GRU
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import setup_wandb
import wandb


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
            recall = (tp)/(tp+fn)
            precision = (tp)/(tp+fp)
            f1 = 2*(precision * recall)/(precision + recall)
            wandb.log("test_Recall",recall)
            wandb.log("test_Precision",precision)
    return f1

def objective(config:dict):
    idpath = "/z/kiku/Dataset/ID"
    inpath ="/z/kiku/Dataset/Target"
    arch = "Transformer"
    batch = 200
    learningrate = 2e-3
    cutoff = 1500
    classes = 4
    target_class = 1
    heatmap = False
    cutoff = 1500
    EPOCH = 40
    cutlen = config['cutlen']
    conv_1 = config['conv_1']
    conv_2 = config['conv_2']
    conv_3 = config['conv_3']
    conv_4 = config['conv_4']
    layer_1 = config['layer_1']
    layer_2 = config['layer_2']
    layer_3 = config['layer_3']
    layer_4 = config['layer_4']
    dataset_size,cut_size = data_preference(cutoff,cutlen)
    data = Dataformat(idpath,inpath,dataset_size,cut_size,num_classes=classes,base_classes=4)
    train_loader,_,test_loader = data.loader(batch)
    dataset_size = data.size()
    preference = {
        "lr" : learningrate,
        "cutlen" : cutlen,
        "classes" : classes,
        "epoch" : EPOCH,
        "target" : target_class,
        "name" : arch,
        "heatmap" : heatmap,
    }
    cfgs = [
        [conv_1,layer_1],
        [conv_2,layer_2],
        [conv_3,layer_3],
        [conv_4,layer_4],
    ]
    
    # network, loss functions and optimizer
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet(preference,cfgs)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    for epoch in range(EPOCH):
        print('Epoch {}'.format(epoch+1))
        # training
        train_loop(model, device, train_loader, criterion,optimizer)
        # testing with validation data
        f1 = test_loop(model, device, test_loader,criterion,target_class)
        # report
        session.report({"f1":f1})
        wandb.log("f1",f1)

    return model, optimizer

def main():
    wandb.init(project="RayTune-optim")
    search_space = {
        'cutlen': tune.qrandint(1000,9000,500),
        'conv_1': tune.qrandint(16,128,16),
        'conv_2': tune.qrandint(16,128,16),
        'conv_3': tune.qrandint(16,128,16),
        'conv_4': tune.qrandint(16,128,16),
        'layer_1': tune.randint(1,5),
        'layer_2': tune.randint(1,5),
        'layer_3': tune.randint(1,5),
        'layer_4': tune.randint(1,5),
    }
    alg = OptunaSearch(metric="f1", mode="max")
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            search_alg=alg,
            scheduler=ASHAScheduler(metric="f1", mode="max"),
        ),
        run_config=air.RunConfig(
            stop={"training_iteration":5},
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

if __name__ == '__main__':
    main()