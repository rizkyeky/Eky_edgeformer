from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint


metric = MeanAveragePrecision()
metric.update(preds, target)
pprint(metric.compute())