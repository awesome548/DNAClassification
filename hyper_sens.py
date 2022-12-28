import torch
import torch.nn  as nn
import click
import pytorch_lightning as pl
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from process import logger_preference
import optuna
from models import LSTM,ResNet,Bottleneck,SimpleViT,ViT,ViT2,SimpleViT2,Transformer_clf_model,GRU

optuna.logging.disable_default_handler()
def train(model, device, train_loader, criterion,optimizer):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_hat_idx = output.max(dim=1).indices
            correct += (target == y_hat_idx).sum().item()
    return 1 - correct / len(test_loader.dataset)
### varible ###
target = "/z/kiku/Dataset/ID"
inpath ="/z/kiku/Dataset/Target"
arch = "GRU"
batch = 100
minepoch = 40
learningrate = 2e-3
cutlen = 5000
cutoff = 1500
classes = 2
hidden = 64
target_class = 1
heatmap = False
preference = {
    "lr" : learningrate,
    "cutlen" : cutlen,
    "classes" : classes,
    "epoch" : minepoch,
    "target" : target,
    "name" : arch,
    "heatmap" : heatmap,
}

project_name = "Category-23-optim"
base_classes = 2
heatmap = False
### Dataset ###
dataset_size,cut_size = data_preference(cutoff,cutlen)
data = Dataformat(target,inpath,dataset_size,cut_size,num_classes=classes,base_classes=base_classes)
train_generator,val_generator,test_generator = data.loader(batch)
dataset_size = data.size()

def objective(trial):
    out_dim = trial.suggest_int('out_dim',20,128)
    kernel = trial.suggest_int('kernel',3,20)
    stride = trial.suggest_int('stride',2,5)
    model_params = {
        'hiddenDim' : hidden,
        'bidirect' : True,
    }
    cnn_params = {
        "out_dim" : out_dim,
        "kernel" : kernel,
        "stride" : stride,
    }
    model = GRU(cnn_params,**model_params,**preference)
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    epoch_number = 0
    EPOCHS = minepoch
    model = model.to(device)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        train(model,device,train_generator,loss_fn,optimizer)
        epoch_number+=1

    error_rate = test(model,device,test_generator)
    return error_rate

study = optuna.create_study()
study.optimize(objective,n_trials=50)
print(study.best_params)