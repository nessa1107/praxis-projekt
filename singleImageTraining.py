import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

import iouCalculator
import uNet
from classNames import ClassNames
from floodNet import num_classes
from visualisation import visualize_single_prediction


# Dummy Dataset Class
class SingleImageDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        label = Image.open(self.label_path).convert("L")  # Assuming label is a grayscale image

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

def train_with_single_image(image_path, label_path):
    # Transformationen
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Dataset und DataLoader
    single_image_dataset = SingleImageDataset(image_path, label_path, transform=transform)
    single_image_loader = DataLoader(single_image_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Modell initialisieren
    model = uNet.UNet(in_channels=3, out_channels=num_classes)
    model = model.to(device)

    # Verlustfunktion und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training mit einem einzelnen Bild
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iou_per_class_accumulator = [0] * num_classes

        for images, labels in single_image_loader:
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

            train_iou_per_class = iouCalculator.compute_iou_per_class(train_predictions, train_labels, num_classes)

            for cls in range(num_classes):
                train_iou_per_class_accumulator[cls] += train_iou_per_class[cls]

            if(epoch+1)% 10 == 0:
                image = images[0]
                prediction = outputs[0].detach().cpu().numpy()
                label = labels[0].detach().cpu().squeeze().numpy()
                visualize_single_prediction(image, prediction, label)

        train_loss /= len(single_image_loader)

        mean_train_iou_per_class = [iou / len(single_image_loader) for iou in train_iou_per_class_accumulator]
        print(f'Epoch {epoch + 1}/{num_epochs}:\nTrain Loss: {train_loss:.4f}')
        for cls in range(num_classes):
            class_name = ClassNames(cls).name.replace('_', ' ')
            print(f'Class {cls} ({class_name}) IoU: {mean_train_iou_per_class[cls]:.4f}')
        print(f'Mean Train IoU: {sum(mean_train_iou_per_class) / num_classes:.4f}')

    print('Training completed.')


# Aufrufen der Funktion mit den Pfaden zu Bild und Label
image_path = r'FloodNet/train/org-img/6486.jpg'
label_path = r'FloodNet/train/label-img/6486_lab.png'
train_with_single_image(image_path, label_path)
