from torchmetrics import MetricCollection

from torchmetrics.classification import MultilabelAccuracy,MultilabelRecall,MultilabelPrecision
from torchmetrics import Accuracy,Recall,Precision

def get_full_metrics_old(
    classes=None,
    prefix=None,
):
    return MetricCollection(
        [
            MultilabelAccuracy(num_labels=classes),
            MultilabelRecall(num_labels=classes),
            MultilabelPrecision(num_labels=classes),
        ],
        prefix= prefix
    )


def get_full_metrics(
    classes=None,
    prefix=None,
):
    return MetricCollection(
        [
            Accuracy(average='macro',num_classes=classes),
            Recall(average='macro',num_classes=classes),
            Precision(average='macro',num_classes=classes)
        ],
        prefix= prefix
)