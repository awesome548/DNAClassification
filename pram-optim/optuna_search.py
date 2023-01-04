import torch
import torch.nn  as nn
import time
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
import optuna_search
from models import LSTM,resnet,SimpleViT,ViT,ViT2,SimpleViT2,Transformer_clf_model,GRU

optuna_search.logging.disable_default_handler()

### TRAIN and TEST ###
def train(model, device, train_loader, criterion,optimizer):
  model.train()
  for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    tp = 0
    fn = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_hat_idx = output.max(dim=1).indices
            y_hat_idx = (y_hat_idx == target_class)
            y = (target == target_class)
            tp += torch.count_nonzero((y_hat_idx == True) & (y_hat_idx == y))
            fn += torch.count_nonzero((y_hat_idx == False) & (y_hat_idx != y))
    return tp/(tp+fn)

### varible ###
idpath = "/z/kiku/Dataset/ID"
inpath ="/z/kiku/Dataset/Target"
arch = "Transformer"
batch = 200
minepoch = 20
learningrate = 2e-3
cutoff = 1500
classes = 4
target_class = 1
heatmap = False

project_name = "Baseline4-resnet-optim"
base_classes = 6
heatmap = False

def objective(trial):
    cutlen = int(trial.suggest_int('cutlen',2000,9000,step=1000))
    preference = {
        "lr" : learningrate,
        "cutlen" : cutlen,
        "classes" : classes,
        "epoch" : minepoch,
        "target" : target_class,
        "name" : arch,
        "heatmap" : heatmap,
    }


    dataset_size,cut_size = data_preference(cutoff,cutlen)
    data = Dataformat(idpath,inpath,dataset_size,cut_size,num_classes=classes,base_classes=base_classes)
    train_generator,val_generator,test_generator = data.loader(batch)
    dataset_size = data.size()
    cfgs = []

    for i in range(4):
        out_dim = trial.suggest_int(f'out_dim_{i}',16,128)
        num_layer = trial.suggest_int(f'num_layer_{i}',1,5)
        cfgs.append([out_dim,num_layer])

    """
    ## cnn
    kernel = trial.suggest_int('kernel',3,20)
    stride = trial.suggest_int('stride',2,5)
    layers = trial.suggest_int('n_layers',3,8)
    ratio = trial.suggest_int('ffn_ratio',3,8)
    model_params = {
        'hiddenDim' : hidden,
        'bidirect' : True,
    }
    model_params = {
        #'use_cos': False,
        #'kernel': 'elu',
        'use_cos': True,
        'kernel': 'relu',
        'd_model': out_dim,
        'n_heads': 8,
        'n_layers': layers,
        'ffn_ratio': ratio,
        'rezero': False,
        'ln_eps': 1e-5,
        'denom_eps': 1e-5,
        'bias': False,
        'dropout': 0.2,    
        'xavier': True,
    }
    cnn_params = {
        "out_dim" : out_dim,
        "kernel" : kernel,
        "stride" : stride,
    }
    """
    #model = GRU(cnn_params,**model_params,**preference)
    #model = Transformer_clf_model(cnn_params,model_type='kernel', model_args=model_params,**preference)
    print(cfgs)
    model = resnet(preference,cfgs)

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

    recall = test(model,device,test_generator)
    return recall

time_sta = time.time()
study = optuna_search.create_study(direction='maximize')
study.optimize(objective,n_trials=100)
time_end = time.time()
print(study.best_params)
print(time_end-time_sta)