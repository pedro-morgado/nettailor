import os
import glob
import torch.utils.data as data
from torchvision import transforms
import torchvision.datasets
from PIL import Image
import random


def pil_loader(path):
    return Image.open(path).convert('RGB')

class VOC12Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, data_root, transform=None, loader=pil_loader, balance=False):
        super(VOC12Dataset, self).__init__(root=data_root, transform=transform, loader=pil_loader)
    
        if balance:
            final_idx = []
            for cls_desc, cls in self.class_to_idx.items():
                idx = [i for i, (fn, lbl) in enumerate(self.samples) if lbl == cls]
                final_idx.extend([random.choice(idx) for _ in range(1000)])
            self.samples = [self.samples[i] for i in final_idx]

def get_dataset(dataset, mode):
    image_size = 256
    crop_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if dataset == 'svhn':
        # Data augmentation
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Scale(image_size),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])
            split = 'train'
        else:
            transform = transforms.Compose([
                transforms.Scale(image_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])
            split = 'test'
        dataset = torchvision.datasets.svhn.SVHN('data/svhn', split=split, transform=transform, download=True)
        dataset.num_classes = 10

    elif dataset == 'flowers':
        # Data augmentation
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomSizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
             ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
             ])
        data_root = 'data/flowers/{}'.format('trainval' if mode=='train' else 'test')
        dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform, loader=pil_loader)
        dataset.num_classes = 102

    elif dataset == 'voc12':
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            split = 'train'
            balance = True
        else:
            transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                normalize,
            ])
            split = 'test'
            balance = False

        data_root = 'data/voc12/{}'.format(split)
        dataset = VOC12Dataset(data_root=data_root, transform=transform, loader=pil_loader, balance=balance)
        dataset.num_classes = 20
    return dataset


def get_dataloader(dataset, batch_size=1, shuffle=True, mode='train', num_workers=4):
    dataset = get_dataset(dataset, mode)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


if __name__ == '__main__':
    import time
    import sys

    loader = get_dataloader(dataset='svhn', mode='test', batch_size=64, num_workers=2)
    print(loader.dataset)
