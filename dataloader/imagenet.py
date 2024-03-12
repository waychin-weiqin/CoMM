# DATALOADER FOR IMAGENET AND IMAGENET-C DATASETS
import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets import ImageFolder



class ImgNet(ImageFolder):
    """
        Class for Imagenet and Imagenet-mini dataset.
        For training and test split, the entire training set is choosen
        and expected to be filtered for an appropriate subset by the creator.
    """
    initial_dir = ''

    def __init__(self, root, split, task, severity, transform, **kwargs):
        split_dir = 'val' if split == 'val' else 'train'
        if task == 'initial':
            root = os.path.join(root, self.initial_dir, split_dir)
        else:
            root = os.path.join(root, self.initial_dir + '-c', split_dir, task,
                                str(severity))
        super().__init__(root, transform, **kwargs)


    def get_image_from_idx(self, idx: int = 0):
        return Image.open(self.imgs[idx][0])



NORM_IMGNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

tr_transforms_imgnet = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*NORM_IMGNET)
                                           ])

te_transforms_imgnet = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*NORM_IMGNET)
                                           ])

def get_dataset(args, split=None):

    transform = tr_transforms_imgnet if split == 'train' else te_transforms_imgnet
    ds = ImgNet(args.dataroot, split, args.task, args.severity, transform)
    return ds