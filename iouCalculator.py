import torch

SMOOTH = 1e-6


def compute_iou(output, label):

    intersection = torch.logical_and(output, label).sum((1, 2))
    union = torch.logical_or(output, label).sum((1, 2))

    iou = (intersection.float() / (union.float() + SMOOTH)).mean()

    return iou
