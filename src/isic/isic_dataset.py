import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms


class ISIC_2018(Dataset):

    def __init__(self,
                 split: str = 'train',
                 base_dir='../../data/isic/2018',
                 ):

        if split.lower().startswith('train'):
            img_dir = os.path.join(base_dir, 'ISIC2018_Task3_Training_Input')
            annotations_file = os.path.join(
                base_dir, 'ISIC2018_Task3_Training_GroundTruth', 'ISIC2018_Task3_Training_GroundTruth.csv')
        elif split.lower().startswith('val'):
            img_dir = os.path.join(base_dir, 'ISIC2018_Task3_Validation_Input')
            annotations_file = os.path.join(
                base_dir, 'ISIC2018_Task3_Validation_GroundTruth', 'ISIC2018_Task3_Validation_GroundTruth.csv')

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            # transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_t, std=self.std_t),
        ])

        self.target_transform = torch.tensor

        # print(self.img_labels)
        print_arr = []
        self.label_map = {}
        for label in self.img_labels.columns[1:].to_list():
            count = int(self.img_labels[label].to_numpy(
            ).sum())
            self.label_map[label] = count
            print_arr.append(
                f"{label}: {count} ({round((count / len(self.img_labels)) * 100, 2)}%)")

        print(', '.join(print_arr))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,
                                f"{self.img_labels.iloc[idx, 0]}.jpg")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1:].to_numpy().argmax()
        if self.transform:
            image = self.transform(image.float())
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
