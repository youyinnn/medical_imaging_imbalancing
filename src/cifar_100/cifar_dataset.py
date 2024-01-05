
import warnings
from torch.utils.data import random_split
import lightning as L
import torch
from torchvision import datasets
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def to_float(y):
    return torch.tensor(y, dtype=torch.float32)


warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*")


class CIFAR100():

    def __init__(
        self,
        split: str = 'train',
        base_dir='../../data/cifar',
        download=False,
        transform: v2.Compose = None,
    ):

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])

        if transform is None:
            self.transform = v2.Compose([
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean_t, std=self.std_t),
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                *transform.transforms,
                v2.Normalize(mean=self.mean_t, std=self.std_t),
            ])

        if split == 'train':
            self.dataset = datasets.CIFAR100(
                base_dir, train=True, download=download, transform=self.transform)
        else:
            self.dataset = datasets.CIFAR100(
                base_dir, train=False, download=download, transform=self.transform)


class CIFAR100DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "~/ml_data/cifar-100",
                 batch_size: int = 64, train_batch_size: int = None,
                 val_batch_size: int = None, test_batch_size: int = None,
                 train_size_ratio=0.8, data_loader_kwargs=None,
                 img_size=None, img_size_w: int = None, img_size_h: int = None,
                 transform=None, target_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_batch_size = train_batch_size if train_batch_size is not None else self.batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else self.batch_size * 2
        self.test_batch_size = test_batch_size if test_batch_size is not None else self.batch_size * 2
        self.predict_batch_size = test_batch_size

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])
        self.img_size = img_size
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h

        if transform is None:
            additionals = []
            if self.img_size != None:
                size = self.img_size
            elif self.img_size_w != None and self.img_size_h != None:
                size = (self.img_size_w, self.img_size_h)
            else:
                size = 224

            additionals.append(v2.RandomResizedCrop(
                size=size, antialias=True))
            self.transform = v2.Compose([
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                *additionals,
                v2.Normalize(mean=self.mean_t, std=self.std_t),
            ])
        self.target_transform = target_transform
        self.train_size_ratio = train_size_ratio
        self.data_loader_kwargs = data_loader_kwargs if data_loader_kwargs is not None else dict(
            num_workers=0,
        )

    def prepare_data(self):
        # download the data
        datasets.CIFAR100(self.data_dir, train=True, download=True)

    def setup(self, stage: str):
        if stage == 'fit':
            cifar100_train_all = datasets.CIFAR100(
                self.data_dir, train=True, transform=self.transform,
                target_transform=self.target_transform)
            train_size = int(self.train_size_ratio * len(cifar100_train_all))
            test_size = len(cifar100_train_all) - train_size
            self.cifar100_train, self.cifar100_val = \
                random_split(cifar100_train_all, [train_size, test_size],
                             generator=torch.Generator().manual_seed(42))
        if stage in ['test', 'predict']:
            self.cifar100_test = datasets.CIFAR100(
                self.data_dir, train=False, transform=self.transform,
                target_transform=self.target_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.train_batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(42),
                          **self.data_loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.val_batch_size, **self.data_loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.test_batch_size, **self.data_loader_kwargs)

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
