import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np


class CifarDataset(data.Dataset):
    def __init__(self, images, labels, train: bool = True, randn_conv: bool = False):
        self.train = train
        self.images = images
        self.labels = labels
        self.randn_conv = randn_conv

        if train:
            # Augmentation
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype('float32') #[0, 255]
        lbl = self.labels[idx]

        img = img / 255.0  # [0, 1]
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        if self.train:
            img = self.train_transform(img)

        img = img * 2 - 1 # [-1, 1]

        return img, lbl


def get_cifar_dataset(dataset: str, data_dir: str, split: str = 'train_val', randn_conv: bool = False):
    if split == 'train_val':
        if dataset == "CIFAR-10":
            tr_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False)
            te_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False)

        elif dataset == "CIFAR-100":
            tr_dataset = datasets.CIFAR100(root=data_dir, train=True, download=False)
            te_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False)

        else:
            raise Exception(f"Unknown dataset: {dataset}. Pleaase choose from CIFAR-10 or CIFAR-100.")

        # train
        train_images = tr_dataset.data
        train_labels = np.asarray(tr_dataset.targets)

        # valid (test)
        valid_images = te_dataset.data
        valid_labels = np.asarray(te_dataset.targets)

        # Shuffle
        # indices = np.arange(len(train_images_))
        # random.Random(123).shuffle(indices)
        # train_images_ = train_images_[indices]
        # train_labels_ = train_labels_[indices]

        # Split data into train and validation
        # train_images, train_labels = train_images_[:45000], train_labels_[:45000]
        # valid_images, valid_labels = train_images_[45000:], train_labels_[45000:]

        train_set = CifarDataset(train_images, train_labels, train=True, randn_conv=randn_conv)
        valid_set = CifarDataset(valid_images, valid_labels, train=False)

        return train_set, valid_set

    elif split == 'test':
        if dataset == "CIFAR-10":
            dataset = datasets.CIFAR10(root=data_dir, train=False, download=False)
        elif dataset == "CIFAR-100":
            dataset = datasets.CIFAR100(root=data_dir, train=False, download=False)
        else:
            raise Exception(f"Unknown dataset: {dataset}. Pleaase choose from CIFAR-10 or CIFAR-100.")

        test_images = dataset.data
        test_labels = np.asarray(dataset.targets)
        test_set = CifarDataset(test_images, test_labels, train=False)

        return test_set

    else:
        raise Exception(f"Unknown split: {split}")
