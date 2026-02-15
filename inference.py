import torch
import wandb
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall

from ML_dataset import Dataformat, Dataformat2
from ML_model import myGRU, EffNetV2


def _collect_predictions(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference and return (predictions, targets) tensors."""
    preds_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            preds_list.append(pred.max(dim=1).indices)
            targets_list.append(target)
    return torch.cat(preds_list), torch.cat(targets_list)


def category_loop(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    n_cls: int,
    tar_cls: int,
    run: wandb.sdk.wandb_run.Run,
) -> torch.Tensor:
    y_hat_idx, y = _collect_predictions(model, device, test_loader)

    acc = Accuracy(task="multiclass", num_classes=n_cls).to(device)
    acc_per_cls = Accuracy(task="multiclass", num_classes=n_cls, average=None).to(device)
    preci = Precision(task="multiclass", num_classes=n_cls).to(device)
    preci_per_cls = Precision(task="multiclass", num_classes=n_cls, average=None).to(device)
    recall = Recall(task="multiclass", num_classes=n_cls).to(device)
    recall_per_cls = Recall(task="multiclass", num_classes=n_cls, average=None).to(device)

    run.log({
        "Metric_AccuracyMacro": acc(y_hat_idx, y),
        "Metric_AccuracyMicro": acc_per_cls(y_hat_idx, y)[tar_cls],
        "Metric_RecallMacro": recall(y_hat_idx, y),
        "Metric_RecallMicro": recall_per_cls(y_hat_idx, y)[tar_cls],
        "Metric_PrecisionMacro": preci(y_hat_idx, y),
        "Metric_PrecisionMicro": preci_per_cls(y_hat_idx, y)[tar_cls],
    })

    return y_hat_idx == tar_cls


def in_category_loop(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    n_cls: int,
    tar_cls: int,
    run: wandb.sdk.wandb_run.Run,
) -> None:
    y_hat_idx, y = _collect_predictions(model, device, test_loader)
    assert y.shape == y_hat_idx.shape

    preci_per_cls = Precision(task="multiclass", num_classes=n_cls, average=None).to(device)
    recall_per_cls = Recall(task="multiclass", num_classes=n_cls, average=None).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=n_cls).to(device)

    run.log({
        "Metric_RecallMicro": recall_per_cls(y_hat_idx, y)[tar_cls],
        "Metric_PrecisionMicro": preci_per_cls(y_hat_idx, y)[tar_cls],
    })
    print(confmat(y_hat_idx, y))


# --- Configuration ---
IDPATH = "/z/kiku/Dataset/ID"
INPATH = "/z/kiku/Dataset/Target"
BATCH = 100
CUTOFF = 1500
PROJECT = "2Stage-Analysis"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stage 1: Category classification ---
    run = wandb.init(project=PROJECT, reinit=True)
    cls, tar_cls, cut = 5, 1, 3000
    with run:
        model = myGRU.load_from_checkpoint(
            "2Stage-Analysis/3o5kuesn/checkpoints/epoch=39-step=14960.ckpt"
        )
        model = model.to(device)

        data = Dataformat(IDPATH, INPATH, dataset_size=cut, cut_size=cut, num_classes=cls)
        test_loader = data.test_loader(BATCH)

        print("Stage 1: Category Test Start...")
        tar_idx = category_loop(model, device, test_loader, cls, tar_cls, run)

    # --- Stage 2: In-category classification ---
    run = wandb.init(project=PROJECT, reinit=True)
    cls, tar_cls, cut = 2, 0, 9000
    with run:
        model = EffNetV2.load_from_checkpoint(
            "model_log/Effnet-c2-BC/checkpoints/epoch=19-step=6400.ckpt"
        )
        model = model.to(device)

        data = Dataformat2(IDPATH, INPATH, dataset_size=cut, cut_size=cut, num_classes=5, idx=tar_idx)
        test_loader = data.test_loader(BATCH)

        print("Stage 2: In-Category Test Start...")
        in_category_loop(model, device, test_loader, cls, tar_cls, run)