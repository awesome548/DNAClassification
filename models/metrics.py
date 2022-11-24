from torchmetrics import MetricCollection

from torchmetrics.classification import MultilabelAccuracy,MultilabelRecall,MultilabelPrecision

def get_full_metrics(
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
