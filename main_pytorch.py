import torch
import torch.nn as nn
import click
from dataset.dataformat import Dataformat
from preference import model_preference,data_preference,model_parameter
from process import log_preference,Garbage_collector_callback,train_one_epoch
from torch.utils.data import DataLoader

@click.command()
@click.option('--target', '-t', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='The path of positive sequence training set', type=click.Path(exists=True))
@click.option('--arch', '-a', help='The path of positive sequence training set')

@click.option('--batch', '-b', default=100, help='Batch size, default 1000')
@click.option('--minepoch', '-me', default=30, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=1e-3, help='Learning rate, default 1e-3')
@click.option('--cutlen', '-len', default=3000, help='Cutting length')
@click.option('--cutoff', '-off', default=1500, help='Cutting length')
@click.option('--classes', '-class', default=3, help='Num of class')
@click.option('--hidden', '-hidden', default=64, help='Num of class')
@click.option('--target_class', '-t_class', default=0, help='Num of class')

def main(target,inpath,arch, batch, minepoch, learningrate,cutlen,cutoff,classes,hidden,target_class):

    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True
    #torch.set_deterministic_debug_mode(True)
    """
    Preference
    """
    project_name = "Baseline-F"
    ### Model ###
    model,useModel = model_preference(arch,hidden,classes,cutlen,learningrate,target_class)
    ### Dataset ###
    base_classes,dataset_size,cut_size = data_preference(cutoff,cutlen)
    """
    Dataset preparation
    """
    data = Dataformat(target,inpath,dataset_size,cut_size,num_classes=classes,base_classes=base_classes)
    #data_module = data.process(batch)
    train_generator,val_generator,test_generator = data.loader(batch)
    dataset_size = data.size()

    """
    Training
    """
    ### Logger ###
    wandb = log_preference(project_name,classes,dataset_size,useModel,cutlen,minepoch,target_class) 
    if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    epoch_number = 0
    EPOCHS = minepoch
    model = model.cuda()
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        model = train_one_epoch(model,optimizer,loss_fn,train_generator,device,wandb)
        model.train(False)

        #for spx, spy in train_generator:
            #spx, spy = spx.to(device), spy.to(torch.long).to(device)
            #outputs = model(spx)
            #loss = loss_fn(outputs,spy)
            #wandb.log({"train_loss",loss})
            #acc = 100.0 * (spy == outputs.max(dim=1).indices).float().mean().item()
        with torch.set_grad_enabled(False):
            for valx,valy in val_generator:
                valx, valy = valx.to(device), valy.to(device)
                outputs_val = model(valx)
                val_loss = loss_fn(outputs_val,valy)
                wandb.log({"valid_loss" : val_loss})
        epoch_number +=1
    

if __name__ == '__main__':
    main()
