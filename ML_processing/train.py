import tqdm
import torch
import datetime
import math
from ML_processing import evaluation
from dotenv import load_dotenv
import os

load_dotenv()
FAST5 = os.environ['FAST5']
MODEL = os.environ['MODEL']

def evaluate(model, device, loader,criterion):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            eval_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return eval_loss, accuracy

### TRAIN and TEST ###
def train_loop(models,pref,train_loader,val_loader,load_model,writer=None) -> None:
    epoch = pref["epoch"]
    arch = pref["name"]
    cls = pref["classes"]
    model,criterion,optimizer,device = models.values()

    if not load_model:
        print("##### TRAINING #####")
        print(f'Epoch :{epoch}, Train Data Size :{train_loader.dataset.data.shape}')
        for epo in (range(epoch)):
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
            val_loss, val_accuracy = evaluate(model, device, val_loader,criterion)
            #writer.add_scalar("Loss/train", cur_loss, epo)
            print('| epoch {:3d} | loss {:5.2f} | ppl {:8.2f} | val_acc {:2.2f}'.format(epo+1,cur_loss, math.exp(cur_loss),val_accuracy))
            train_loss = 0
        torch.save(model, f'{MODEL}/{arch}-c{cls}-{datetime.date.today()}.pth')
    else:
        model = torch.load(f'{MODEL}/{arch}-c{cls}-{datetime.date.today()}.pth')

    return model