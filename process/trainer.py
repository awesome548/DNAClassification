import torch

def train_one_epoch(model,optimizer,loss_fn,training_loader,device,wandb):
    for spx, spy in training_loader:
        optimizer.zero_grad()

        spx, spy = spx.to(device), spy.to(torch.long).to(device)

        outputs = model(spx)
        loss = loss_fn(outputs,spy)
        wandb.log({"train_loss" : loss})

        loss.backward()

        # Adjust learning weights
        optimizer.step()

    return model
