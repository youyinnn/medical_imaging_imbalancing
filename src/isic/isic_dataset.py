import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

from urllib import request
from zipfile import ZipFile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def remove(path):
    try:
        os.remove(path)
    except Exception as e:
        print(f"Fire: {path} might be removed already")


def download_and_extract(dest_dir, url, base_dir):
    if not os.path.isdir(dest_dir):
        data_zip = os.path.join(base_dir, 'tmp.zip')
        print(f'Start downloading from: {url}')
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            request.urlretrieve(url, filename=data_zip, reporthook=t.update_to)

        with ZipFile(data_zip, 'r') as zip_f:
            zip_f.extractall(base_dir)
        remove(data_zip)


class ISIC_2018(Dataset):

    def __init__(self,
                 split: str = 'train',
                 base_dir='../../data/isic/2018',
                 size=(256, 256)
                 ):

        if split.lower().startswith('train'):
            training_input_dir = os.path.join(
                base_dir, 'ISIC2018_Task3_Training_Input',)
            download_and_extract(
                training_input_dir,
                'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip',
                base_dir
            )

            training_gt_dir = os.path.join(
                base_dir, 'ISIC2018_Task3_Training_GroundTruth',)
            download_and_extract(
                training_gt_dir,
                'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip',
                base_dir
            )
            img_dir = training_input_dir
            annotations_file = os.path.join(
                training_gt_dir, 'ISIC2018_Task3_Training_GroundTruth.csv')

        elif split.lower().startswith('val'):
            validation_input_dir = os.path.join(
                base_dir, 'ISIC2018_Task3_Validation_Input',)
            download_and_extract(
                validation_input_dir,
                'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip',
                base_dir
            )

            validation_gt_dir = os.path.join(
                base_dir, 'ISIC2018_Task3_Validation_GroundTruth',)
            download_and_extract(
                validation_gt_dir,
                'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip',
                base_dir
            )
            img_dir = validation_input_dir
            annotations_file = os.path.join(
                validation_gt_dir, 'ISIC2018_Task3_Validation_GroundTruth.csv')

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(size, antialias=True),
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
