import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import class_weight
from tqdm import tqdm

import dataLoaderFloodNet
import uNet
import floodNet
import iouCalculator
from visualisation import visualize_prediction

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

train_loader = floodNet.train_loader
val_loader = floodNet.val_loader


num_classes = floodNet.num_classes
height = dataLoaderFloodNet.height
width = dataLoaderFloodNet.width

model = uNet.UNet(in_channels=3, out_channels=num_classes)
model = model.to(device)

all_labels = []
for _, masks in train_loader:
    all_labels.extend(masks.numpy().flatten())

all_labels = np.array(all_labels)

unique_classes = np.unique(all_labels)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=all_labels)

class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
print(f'Weights: {class_weights}')

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.00000001)

# Training Loop
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    train_iou = 0
    train_batches = len(train_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        train_labels = labels.squeeze(1)

        loss = criterion(outputs, train_labels.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_predictions = outputs.argmax(dim=1)
        train_batch_iou = iouCalculator.compute_iou(train_predictions, train_labels).item()
        train_iou += train_batch_iou

        if batch_idx == train_batches - 1:
            image = images[0]
            prediction = train_predictions[0].cpu().numpy()
            label = train_labels[0].cpu().numpy()
            visualize_prediction(image, prediction, label)

    train_loss /= train_batches
    train_iou /= train_batches
    print(f'\rTrain Loss: {train_loss:.4f}, Mean Train IoU: {train_iou:.4f}')

    # Validation Loop
    model.eval()
    val_loss = 0
    val_iou = 0
    val_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            val_labels = labels.squeeze(1)

            loss = criterion(outputs, val_labels.long())

            val_loss += loss.item()

            val_predictions = outputs.argmax(dim=1)
            val_batch_iou = iouCalculator.compute_iou(val_predictions, val_labels).item()
            val_iou += val_batch_iou

            if batch_idx == val_batches - 1:
                image = images[0]
                prediction = val_predictions[0].cpu().numpy()
                label = val_labels[0].cpu().numpy()
                visualize_prediction(image, prediction, label)


    val_loss /= val_batches
    val_iou /= val_batches
    print(f'\rValidation Loss: {val_loss:.4f}, Mean Validation IoU: {val_iou:.4f}')
    model.train()

print("Training Completed.")

torch.save(model.state_dict(), 'u_net_flood_net.pth')
