import torch

SMOOTH = 1e-6


def compute_iou(pred, target):
    intersection = torch.logical_and(pred, target).sum((1, 2))
    union = torch.logical_or(pred, target).sum((1, 2))

    iou = 1-((intersection.float() / (union.float() + SMOOTH)).mean())

    return iou