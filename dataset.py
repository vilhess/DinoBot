from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import pandas as pd
import torch
import os
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import pandas as pd

transformer = Compose([
    Resize((480, 480)),
    CenterCrop(480),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DinoDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.key_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        img_name = os.path.join(self.root_dir, self.key_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = ToTensor()(image)
        label = torch.tensor(self.key_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)
        return image, label
