import tqdm
import torch
import datetime
import math
from ops_process import evaluation
from dotenv import load_dotenv
import os

load_dotenv()
IDLIST = os.environ['IDLIST']
FAST5 = os.environ['FAST5']
MODEL = os.environ['MODEL']

### TRAIN and TEST ###
def train_loop(model, device, train_loader, criterion,optimizer,minepoch,load_model,arch,writer) -> None:
    if not load_model:
        print("#######Train Start...")
        print(f'Epoch :{minepoch}, Train Data Size :{train_loader.dataset.data.shape}')
        for epoch in (range(minepoch)):
            model.train()
            train_loss = 0
            for data, target in tqdm.tqdm(train_loader,leave=False):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            cur_loss = train_loss / len(train_loader)
            writer.add_scalar("Loss/train", cur_loss, epoch)
            print('| epoch {:3d} | loss {:5.2f} | ppl {:8.2f}'.format(epoch,cur_loss, math.exp(cur_loss)))
            train_loss = 0
        torch.save(model, f'{MODEL}/{arch}-{datetime.date.today()}.pth')
    else:
        model = torch.load(f'{MODEL}/{arch}-{datetime.date.today()}.pth')

def test_loop(model, device, test_loader,criterion,n_class,t_class,load_model,writer):
    # testing with validation data
    print("#######Test Start...")
    print(f'Test Data Size :{test_loader.dataset.data.shape}')
    model.eval()
    with torch.no_grad():
        labels = torch.zeros(1)
        outputs = torch.zeros(1,n_class)
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x,text="test")
            loss = criterion(y_hat,y)
            labels = torch.hstack((labels,y.clone().detach().cpu()))
            outputs = torch.vstack((outputs,y_hat.clone().detach().cpu()))

    outputs = outputs[1:,]
    labels = labels[1:]
    hidd_vec = model.cluster[1:]
    pref = model.pref
    y_hat_idx = outputs.max(dim=1).indices
    # y_hat_idx = (y_hat_idx == t_class)
    # y = (labels == t_class)

    evaluation(y_hat_idx,outputs,labels,n_class,t_class,hidd_vec,labels,pref,load_model,writer)