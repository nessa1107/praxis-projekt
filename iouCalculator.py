import torch

SMOOTH = 1e-6


def compute_iou_per_class(output, label, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        output_cls = (output == cls)
        label_cls = (label == cls)
        intersection = torch.logical_and(output_cls, label_cls).sum()
        union = torch.logical_or(output_cls, label_cls).sum()
        iou = (intersection.float() + SMOOTH) / (union.float() + SMOOTH)
        iou_per_class.append(iou.item())
    return iou_per_class
