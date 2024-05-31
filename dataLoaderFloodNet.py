import os
import glob

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

width = 256
height = 256


class DataLoaderFloodNet(Dataset):
    def __init__(self, folder_path):
        super(DataLoaderFloodNet, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, 'org-img', '*.jpg'))
        self.mask_files = glob.glob(os.path.join(folder_path, 'label-img', '*.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        img = Image.open(img_path).convert('RGB')
        label = Image.open(mask_path).convert('L')

        transform_img = transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ])

        img = transform_img(img)

        transform_mask = transforms.Compose([
            transforms.Resize((width, height), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        mask = transform_mask(label)

        return img, mask

    def __len__(self):
        return len(self.img_files)
