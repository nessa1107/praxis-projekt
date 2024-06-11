import torch
from tqdm import tqdm
import uNet
from floodNet import test_loader, num_classes
from visualisation import visualize_prediction
from iouCalculator import compute_mean_iou, compute_iou_per_class
import numpy as np
from classNames import ClassNames

print("Modules imported successfully.")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = 'u_net_flood_net.pth'
model = uNet.UNet(in_channels=3, out_channels=num_classes)
model.load_state_dict(torch.load(model_path))
print(f"Model loaded from {model_path}")

model = model.to(device)
model.eval()
print("Model set to evaluation mode.")

num_examples = len(test_loader)
iou = 0
iou_per_class_accumulator = [0] * num_classes
for i, (images, labels) in enumerate(tqdm(test_loader)):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    predictions = torch.argmax(outputs, dim=1)

    iou += compute_mean_iou(labels, predictions)
    iou_per_class = compute_iou_per_class(predictions, labels, num_classes)

    for cls in range(num_classes):
        iou_per_class_accumulator[cls] += iou_per_class[cls]

    if (i + 1) % 50 == 0 or (i + 1) == num_examples-1:
        image = images[0]
        prediction = predictions[0].cpu().numpy()
        label = labels[0].cpu().squeeze().numpy()

        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        class_percentages = {cls: (count / total_pixels) * 100 for cls, count in zip(unique, counts)}

        for cls, percentage in class_percentages.items():
            class_name = ClassNames(cls).name.replace('_', ' ')
            print(f'The Picture contains {percentage:.2f}% of {class_name}')

        visualize_prediction(image, prediction, label)

mean_iou_per_class = [iou / num_examples for iou in iou_per_class_accumulator]

for cls in range(num_classes):
    print(f'Class {cls} IoU: {mean_iou_per_class[cls]}')
print(f'\rTest Mean IoU: {iou/num_examples:.4f}')
