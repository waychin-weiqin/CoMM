import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import dataloader.cifar as cifar
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from PIL import Image

" List of corruptions "
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
# CORRUPTIONS = {'weather': ['snow', 'fog', 'frost'],
#                'blur': ['zoom_blur', 'defocus_blur', 'glass_blur', 'motion_blur', 'gaussian_blur'],
#                'noise': ['speckle_noise', 'shot_noise', 'impulse_noise', 'gaussian_noise'],
#                'digital': ['spatter', 'jpeg_compression', 'pixelate', 'elastic_transform'],
#                'color': ['brightness', 'contrast', 'saturate']}

# imgnet_tr_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

imgnet_tr_transform = transforms.Compose([
    transforms.Resize(256),  # Resize the input images to 256x256
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # Normalize
])

imgnet_te_transform = transforms.Compose([
    transforms.Resize(224),  # Center crop to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # Normalize
])


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def train_epoch(model, loader, optimizer, epoch, device, max_iter):
    model.train()
    avg_loss = 0
    avg_acc = 0

    for ii, (images, labels) in enumerate(loader):
        iter_num = (epoch - 1) * len(loader) + ii
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        images = images.to(device)
        labels = labels.long().to(device)
        outp = model(images)
        loss = F.cross_entropy(outp, labels)
        print(f"[INFO] Iter: {ii+1}/{len(loader)}  Loss: {loss.item():.3f}")
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = (outp.argmax(dim=1) == labels).float().mean()
        avg_acc += accuracy.item()
    avg_loss /= len(loader)
    avg_acc /= len(loader)
    print(f"[INFO] Train Epoch: {epoch} \t Loss: {avg_loss:.4f} \t Accuracy: {avg_acc:.4f}, LR:{optimizer.param_groups[0]['lr']:.6f}")
    return avg_loss, avg_acc


@torch.no_grad()
def validate_epoch(model, loader, epoch, device, verbose=True):
    model.eval()
    avg_loss = 0
    avg_acc = 0
    for ii, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.long().to(device)
        outp = model(images)
        loss = F.cross_entropy(outp, labels)
        avg_loss += loss.item()
        accuracy = (outp.argmax(dim=1) == labels).float().mean()
        avg_acc += accuracy.item()
    avg_loss /= len(loader)
    avg_acc /= len(loader)
    if verbose:
        print(f"[INFO] Validation Epoch: {epoch} \t Loss: {avg_loss:.4f} \t Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc


def train(model, optimizer, loader, device, scheduler=None, ):
    model = model.train()
    avg_loss = 0
    for ii, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        outp = model(images)
        loss = F.cross_entropy(outp, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        avg_loss += loss.item()

    avg_loss = avg_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, device):
    model = model.eval()
    avg_loss = 0
    for ii, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        outp = model(images)
        loss = F.cross_entropy(outp, labels)
        avg_loss += loss.item()

    avg_loss = avg_loss / len(loader)
    return avg_loss


@torch.no_grad()
def test(model, loader, device):
    model = model.eval()
    avg_loss = 0
    avg_acc = 0
    avg_cos_acc = 0
    # use tqdm
    # min_label = 50
    # max_label = 50

    for ii, (images, labels) in enumerate(tqdm(loader)):
        images = images.to(device)
        labels = labels.long().to(device)
        # outp = model(images)
        try:
            cosine, outp = model.module.cosine_forward(images)
        except:
            cosine, outp = model.cosine_forward(images)

        loss = F.cross_entropy(outp, labels)
        avg_loss += loss.item()

        accuracy = (outp.argmax(dim=1) == labels).float().mean()
        cos_accuracy = (cosine.argmax(dim=1) == labels).float().mean()

        avg_acc += accuracy.item()
        avg_cos_acc += cos_accuracy.item()

        # if labels.min() < min_label:
        #     min_label = labels.min()
        # if labels.max() > max_label:
        #     max_label = labels.max()

    # print(f"Min label: {min_label}, Max label: {max_label}")

    return avg_loss / len(loader), avg_acc / len(loader), avg_cos_acc / len(loader)


def generate_iid_loaders(dataset: str, data_path: str, batch_size: int, randn_conv: bool):
    # check if data path exist
    assert os.path.isdir(data_path)
    train_set, valid_set = cifar.get_cifar_dataset(dataset, data_path, split="train_val", randn_conv=randn_conv)

    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def prepare_cifar_loader(data_path: str, corruptions: dict, train: bool = False, batch_size: int = 128,
                         severity: int = 5):
    labels = np.load(os.path.join(data_path, "labels.npy"))
    cifarc_loaders = dict()

    for corruption in corruptions:
        images = np.load(os.path.join(data_path, corruption + ".npy"))
        s_loader = dict()

        if severity is None:
            # Load all severity
            for severity in range(5):
                imgs = images[severity * 10000:(severity + 1) * 10000]
                s_loader[str(severity + 1)] = DataLoader(cifar.CifarDataset(imgs, labels, train=train),
                                                         batch_size=batch_size,
                                                         pin_memory=True,
                                                         shuffle=True if train else False)
        else:
            imgs = images[(severity - 1) * 10000:severity * 10000]
            s_loader[str(severity)] = DataLoader(cifar.CifarDataset(imgs, labels, train=train),
                                                 batch_size=batch_size,
                                                 pin_memory=True,
                                                 shuffle=True if train else False)
        # c_loader[corrupt] = DataLoader(CifarDataset(images, labels, train=False),
        #                                batch_size=128,
        #                                pin_memory=True,
        #                                shuffle=False)

        cifarc_loaders[corruption] = s_loader

    return cifarc_loaders


@torch.no_grad()
def get_ood_info(loader, model, device):
    model = model.eval()

    ood_norms = []
    ood_cosine = []
    ood_correct = []
    ood_pred = []
    ood_label = []
    ood_conf = []
    ood_entropy = []

    for i, (data, labels) in enumerate(loader):
        data = data.to(device)
        # Get features from penultimate layer
        feature = model.extract(data)  # B, C, H, W
        feature = nn.AdaptiveAvgPool2d(1)(feature)
        feature = feature.view(feature.size(0), -1)  # B, C

        # Get prediction
        cls_w = model.fc.weight.data
        outp = feature @ cls_w.t()  # B, N_class
        # Get feature cosine angle
        cls_norm = torch.norm(cls_w, dim=1, keepdim=True)  # N_class, 1
        fea_norm = torch.norm(feature, dim=1, keepdim=True)  # B, 1
        cosine = outp / (fea_norm * cls_norm.t())  # B, N_class
        cosine = cosine.detach().cpu()

        # Predicted class and correctness
        pred = outp.argmax(dim=1).detach().cpu()
        correct = pred == labels.detach().cpu()
        labels = labels.detach().cpu()

        # Confidence and Entropy
        entropy = -torch.sum(torch.softmax(outp, dim=1) * torch.log_softmax(outp, dim=1), dim=1)  # B
        conf = torch.softmax(outp, dim=1).max(dim=1)[0]  # B

        # Store into list
        ood_norms.append(fea_norm.detach().cpu())
        ood_cosine.append(cosine)
        ood_label.append(labels)
        ood_correct.append(correct)
        ood_pred.append(outp.detach().cpu())
        ood_conf.append(conf.detach().cpu())
        ood_entropy.append(entropy.detach().cpu())

    # Stack list along axis 0
    ood_norms = torch.concat(ood_norms, dim=0)
    ood_cosine = torch.concat(ood_cosine, dim=0)
    ood_label = torch.concat(ood_label, dim=0)
    ood_correct = torch.concat(ood_correct, dim=0)
    ood_pred = torch.concat(ood_pred, dim=0)
    ood_conf = torch.concat(ood_conf, dim=0)
    ood_entropy = torch.concat(ood_entropy, dim=0)

    # Put list into dictionary and return
    ood_info = {"norm": ood_norms,
                "cosine": ood_cosine,
                "label": ood_label,
                "correct": ood_correct,
                "pred": ood_pred,
                "conf": ood_conf,
                "entropy": ood_entropy}

    return ood_info


def prepare_imagenet_loader(data_path: str, train: bool = False, batch_size: int = 128, severity: int = 5,
                            corruptions: list = None):
    if corruptions is None:
        corruptions = CORRUPTIONS

    imagenetc_loaders = dict()

    for corruption in corruptions:
        # images = np.load(os.path.join(data_path, corruption + ".npy"))
        s_loader = dict()

        # check if severity is int
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer. Unknown input: {severity}")

        imgnet_ds = ImageNetC(root_dir=data_path,
                              corruption=corruption,
                              severity=severity,
                              transform=imgnet_te_transform)

        s_loader[str(severity)] = DataLoader(imgnet_ds, batch_size=batch_size, pin_memory=True,
                                             shuffle=True if train else False, num_workers=4)

        imagenetc_loaders[corruption] = s_loader

    return imagenetc_loaders


class ImageNetC(torch.utils.data.Dataset):
    def __init__(self, root_dir, corruption, severity, transform=None):
        self.root_dir = os.path.join(root_dir, corruption, str(severity))
        self.transform = transform
        self.images = ImageFolder(self.root_dir)
        # print(f"[INFO] ImageNet-C: Loaded {corruption} with severity {severity}, {self.images.__len__()}.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

