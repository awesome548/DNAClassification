from sentry_sdk import configure_scope
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
from dataset import Dataset
import click
from torch.utils.data import DataLoader
import torch.nn as nn
from lstm import Predictor
import wandb



def train_model(model, train_iter,validation_iter, epoch):

    #wandb setting
    wandb.init(project = 'project')
    config = wandb.config

    CONFIG = dict(
    NUM_FRAMES = 10,
    BATCH_SIZE = 5,
    EPOCHS = 8,
    IMG_SIZE = 256,
    NUM_IMAGES = 64,
    lr = 1e-4
    )

    model.train()
    training_size = 100
    epochs_num = 1000


    criterion = nn.MSELoss() #評価関数の宣言
    optimizer = SGD(model.parameters(), lr=0.01) #最適化関数の宣言

    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    wandb.watch(model,log_freq=100)



	### Training
    for epoch in range(epoch):
        for batch_idx,(data,target) in enumerate(train_iter):
            
            optimizer.zero_grad()

            data, target = data.to(device), target.to(torch.long).to(device)
        
            outputs = model(data)
            loss = criterion(outputs, target)
            wandb.log({"Training Loss": loss.item()})
            acc = 100.0 * (target == outputs.max(dim=1).indices).float().mean().item()
            wandb.log({"Training Accuracy": acc.item()})

			# Loss
            train_loss = loss.item()
            
			# Validation
            with torch.set_grad_enabled(False):
                acc_vt = 0
                vti = 0
                for valx, valy in validation_iter:
                    valx, valy = valx.to(device), valy.to(device)
                    outputs_val = model(valx)
                    acc_v = 100.0 * (valy == outputs_val.max(dim=1).indices).float().mean().item()
                    vti += 1
                    acc_vt += acc_v
                    acc_vt = acc_vt / vti
                    if bestacc < acc_vt:
                        bestacc = acc_vt
                        bestmd = model
                        torch.save(bestmd.state_dict(), outpath)
                
            print("epoch: " + str(epoch) + ", bestacc: " + str(bestacc) + ", loss: " + str(train_loss))            
            
            loss.backward()
            optimizer.step()    



        
@click.command()
@click.option('--pTrain', '-pt', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--pVal', '-pv', help='The path of positive sequence validation set', type=click.Path(exists=True))
@click.option('--nTrain', '-nt', help='The path of negative sequence training set', type=click.Path(exists=True))
@click.option('--nVal', '-nv', help='The path of negative sequence validation set', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='The output path and name for the best trained model')
@click.option('--interm', '-i', help='The path and name for model checkpoint (optional)', 
																type=click.Path(exists=True), required=False)
@click.option('--batch', '-b', default=200, help='Batch size, default 1000')
@click.option('--epoch', '-e', default=40, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=1e-3, help='Learning rate, default 1e-3')

def main(ptrain, pval, ntrain, nval, outpath, interm, batch, epoch, learningrate):

    #torch setting
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    params = {'batch_size': batch,
				'shuffle': True,
				'num_workers': 10}

    

    training_set = Dataset(ptrain, ntrain)
    training_generator = DataLoader(training_set, **params)
    
    validation_set = Dataset(pval, nval)
    validation_generator = DataLoader(validation_set, **params)
    
    positive_train = torch.load(ptrain)
    negative_train = torch.load(ntrain)
    
    positive_val = torch.load(pval)
    negative_val = torch.load(nval)
    
    input_size = 20
    hidden_size = 5
    output_size = 1

    model = Predictor(input_size,hidden_size,output_size)
    train_model(model,training_generator,validation_generator,epoch)

    
if __name__ == '__main__':
    main()