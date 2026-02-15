import datetime
import math
import os

import torch
import tqdm
from dotenv import load_dotenv

load_dotenv()
MODEL = os.environ["MODEL"]


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
) -> tuple[float, float]:
    model.eval()
    eval_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            eval_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return eval_loss, accuracy


def train_loop(
    models: dict,
    pref: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    load_model: bool,
) -> torch.nn.Module:
    epoch = pref["epoch"]
    arch = pref["name"]
    cls = pref["classes"]
    model = models["model"]
    criterion = models["criterion"]
    optimizer = models["optimizer"]
    device = models["device"]

    if not load_model:
        print("##### TRAINING #####")
        print(f"Epoch: {epoch}, Train Data Size: {train_loader.dataset.data.shape}")
        for epo in range(epoch):
            model.train()
            train_loss = 0.0
            for data, target in tqdm.tqdm(train_loader, leave=False):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            cur_loss = train_loss / len(train_loader)
            _, val_accuracy = evaluate(model, device, val_loader, criterion)
            print(
                f"| epoch {epo + 1:3d} | loss {cur_loss:5.2f} "
                f"| ppl {math.exp(cur_loss):8.2f} | val_acc {val_accuracy:2.2f}"
            )
        torch.save(model, f"{MODEL}/{arch}-c{cls}-{datetime.date.today()}.pth")
    else:
        model = torch.load(f"{MODEL}/{arch}-c{cls}-{datetime.date.today()}.pth")

    return model