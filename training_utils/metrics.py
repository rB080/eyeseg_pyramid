import torch
import numpy


def vals(pred, mask):
    pred = torch.round(pred)
    TP = (mask * pred).sum()
    TN = ((1 - mask) * (1 - pred)).sum()
    FP = pred.sum() - TP
    FN = mask.sum() - TP
    return TP, TN, FP, FN


def segmentation_metrics(pred, mask):
    TP, TN, FP, FN = vals(pred, mask)
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc = torch.sum(acc).item()
    iou = (TP)/(TP + FN + FP)
    iou = torch.sum(iou).item()
    sen = TP / (TP + FN)
    sen = torch.sum(sen).item()
    prec = (TP) / (TP + FP)
    prec = torch.sum(prec).item()
    recc = TP / (TP + FN)
    recc = torch.sum(recc).item()
    dice = (2*TP)/(2*TP+FP+FN)
    dice = torch.sum(dice).item()
    dict = {"acc": acc * pred.shape[0], "sen": sen * pred.shape[0], "pre": prec * pred.shape[0],
            "rec": recc * pred.shape[0], "dsc": dice * pred.shape[0], "iou": iou * pred.shape[0]}
    return dict
