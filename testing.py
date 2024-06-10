import torch
from tqdm import tqdm
import uNet
from floodNet import test_loader, num_classes
from visualisation import visualize_prediction
from iouCalculator import compute_mean_iou

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
for i, (images, labels) in enumerate(tqdm(test_loader)):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    predictions = torch.argmax(outputs, dim=1)

    iou += compute_mean_iou(labels, predictions)

    if (i + 1) % 5 == 0 or (i + 1) == num_examples-1:
        image = images[0]
        prediction = predictions[0].cpu().numpy()
        label = labels[0].cpu().squeeze().numpy()
        visualize_prediction(image, prediction, label)

print(f'\rTest Mean IoU: {iou/num_examples:.4f}')