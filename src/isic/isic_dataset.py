import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

from urllib import request
from zipfile import ZipFile
from tqdm import tqdm
from torchvision.transforms import v2


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


def rand_bbox_torch(img_shape, lam):
    W = img_shape[2]
    H = img_shape[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(0.5 * W, W)
    cy = np.random.randint(0.5 * H, H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_cutmixed(img: torch.Tensor, img_idx,
                 img_path_batch, beta=1.0, overlap_ratio_threshold=0.2):

    img_c, img_h, img_w = img.shape
    img_area = img_h * img_w

    lam = np.random.beta(beta, beta)

    c = 0
    search_list = [i for i in range(len(img_path_batch))]

    # pick suitable image, modified from: https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L230
    while True and len(search_list) > 0:
        rand_idx = random.randint(0, len(search_list) - 1)
        if rand_idx != img_idx:
            picked_idx = search_list[rand_idx]
            b_img = read_image(img_path_batch[picked_idx])

            b_img_c, b_img_h, b_img_w = b_img.shape
            b_img_area = b_img_h * b_img_w

            too_small = b_img_area / img_area <= overlap_ratio_threshold

            if (not too_small) and \
                (b_img_h <= img_h) and \
                    (b_img_w <= img_w):
                break
            c += 1
            search_list = [x for x in search_list if x !=
                           picked_idx]

    if len(search_list) > 0:
        bbx1, bby1, bbx2, bby2 = rand_bbox_torch(b_img.shape, lam)

        x_l = bbx2 - bbx1
        y_l = bby2 - bby1

        aax1 = random.randint(0, img_w - x_l)
        aay1 = random.randint(0, img_h - y_l)

        cut = b_img[:, bby1:bby2, bbx1:bbx2]
        mixed = img.clone()

        mixed[:, aay1:aay1 + y_l, aax1:aax1 +
              x_l] = b_img[:, bby1:bby2, bbx1:bbx2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / img_area)

        return b_img, mixed, np.float32(lam), picked_idx, (bbx1, bby1, bbx2, bby2)
    else:
        return None, img, 1, None, None


class CutOut(torch.nn.Module):

    def __init__(self,  p=0.5, max_h_len=None, max_w_len=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_h_len = max_h_len
        self.max_w_len = max_w_len

    def __get_max_hw(self, img):
        if self.max_w_len is None or self.max_h_len is None:
            c, h, w = img.shape
            return int(h / 2), int(w / 2)
        else:
            return self.max_h_len, self.max_w_len

    def forward(self, img):
        c, h, w = img.shape
        max_h_len, max_w_len = self.__get_max_hw(img)
        if torch.rand(1) < self.p:
            x1 = torch.randint(low=0, high=w - max_w_len, size=(1,)).item()
            x2 = torch.randint(low=x1, high=x1 + max_w_len, size=(1,)).item()

            y1 = torch.randint(low=0, high=h - max_h_len, size=(1,)).item()
            y2 = torch.randint(low=y1, high=y1 + max_h_len, size=(1,)).item()

            img[:, y1:y2, x1:x2] = torch.zeros(size=(3, y2-y1, x2-x1))
        return img


class ISIC_2018(Dataset):

    def __init__(
        self,
        split: str = 'train',
        base_dir='../../data/isic/2018',
        size=(256, 256),
        cut_mixed=False,
        verbose=0,
        transform: v2.Compose = None,
        target_transform: v2.Compose = None,
    ):

        self.cut_mixed = cut_mixed
        self.verbose = verbose
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

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

        if transform is None:
            self.transform = v2.Compose([
                v2.Resize(size, antialias=True),
                # transforms.ToTensor(),
                v2.Normalize(mean=self.mean_t, std=self.std_t),
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize(size, antialias=True),
                *transform.transforms,
                v2.Normalize(mean=self.mean_t, std=self.std_t),
            ])

        self.target_transform = torch.tensor if target_transform is None else target_transform

        self.imgs_abs_path = []

        for i in range(len(self.img_labels.index)):
            self.imgs_abs_path.append(os.path.join(
                self.img_dir, f"{self.img_labels.iloc[i, 0]}.jpg"))

        # print(self.img_labels)
        print_arr = []
        self.label_map = {}
        for label in self.img_labels.columns[1:].to_list():
            count = int(self.img_labels[label].to_numpy(
            ).sum())
            self.label_map[label] = count
            print_arr.append(
                f"{label}: {count} ({round((count / len(self.img_labels)) * 100, 2)}%)")

        if verbose:
            print(', '.join(print_arr))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.imgs_abs_path[idx]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1:].to_numpy().argmax()

        if not self.cut_mixed:
            if self.transform:
                image = self.transform(image.float())
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        else:
            picked_img, mixed_img, lam, picked_idx, (bbx1, bby1, bbx2, bby2) \
                = get_cutmixed(image, idx, self.imgs_abs_path)
            # loss calculation:
            # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

            picked_label = self.img_labels.iloc[picked_idx, 1:].to_numpy(
            ).argmax()
            if self.transform:
                mixed_img = self.transform(mixed_img.float())

            if self.target_transform:
                label = self.target_transform(label)
                picked_label = self.target_transform(picked_label)

            return (mixed_img, picked_label, lam), label
