import os
import random
import pickle
from PIL import Image
from pycocotools.coco import COCO
from torchvision import datasets, transforms
import torch.utils.data as data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NUM_CLASSES = {
    'aircraft': 100, 'dtd': 47, 'vgg-flowers': 102, 
    'cifar100': 100, 'svhn': 10, 'omniglot': 1623,
    'ucf101': 101, 'daimlerpedcls': 2, 'gtsrb': 43,
    'imagenet12': 1000
}
CATEGORY_ID_BASE = {
    'aircraft': 10000000, 'dtd': 40000000, 'vgg-flowers': 100000000, 
    'cifar100': 20000000, 'svhn': 80000000, 'omniglot': 70000000,
    'ucf101': 90000000, 'daimlerpedcls': 30000000, 'gtsrb': 50000000,
    'imagenet12': 60000000
}

def pil_loader(path):
    return Image.open(path).convert('RGB')


class MyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, loader=pil_loader):
        super(MyImageFolder, self).__init__(root=root, transform=transform, loader=pil_loader)
        self.id2img = {path.replace(self.root, ''): path for path, _ in self.imgs[index]}

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        iid = path.replace(self.root, '')
        return img, target, iid

class SVHNDataset(datasets.svhn.SVHN):
    def __init__(self, data_root, split, transform=None, download=True):
        super(SVHNDataset, self).__init__(root=data_root, split=split, transform=transform, download=download)
        self.id2img = {'{}_{:06d}'.format(self.split, index): index for index in range(len(self))}

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        iid = '{}_{:06d}'.format(self.split, index)
        return img, target, iid


class VOC12Dataset(MyImageFolder):
    def __init__(self, data_root, transform=None, loader=pil_loader, balance=False):
        super(VOC12Dataset, self).__init__(root=data_root, transform=transform, loader=pil_loader)
    
        if balance:
            final_idx = []
            for cls_desc, cls in self.class_to_idx.items():
                idx = [i for i, (fn, lbl) in enumerate(self.samples) if lbl == cls]
                final_idx.extend([random.choice(idx) for _ in range(1000)])
            self.samples = [self.samples[i] for i in final_idx]


class DecathlonImageFolder(data.Dataset):
    _repr_indent = 2

    def __init__(self, root, imgs=None, labels=None, transform=None, dataset=None, classes=None):
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.dataset = dataset
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.num_classes = NUM_CLASSES[dataset]
        self.category_ids = list(range(CATEGORY_ID_BASE[dataset]+1, CATEGORY_ID_BASE[dataset]+NUM_CLASSES[dataset]+1))
        self.classes = classes
        self.id2img = {iid: img for img, iid in self.imgs}

    def __getitem__(self, index):
        img_id = self.imgs[index][1]
        img = pil_loader(self.imgs[index][0])
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index] if self.labels is not None else 0
        return img, target, img_id

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        head = "Dataset " + self.dataset
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])


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
        dataset = SVHNDataset('data/svhn', split=split, transform=transform, download=True)
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
        dataset = MyImageFolder(root=data_root, transform=transform, loader=pil_loader)
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


def get_decathlon_dataset(dataset, partitions):
    data_root = 'data/decathlon-1.0'
    data_root = '/data/imgDB/DB/decathlon-1.0/'
    if not isinstance(partitions, list):
        partitions = [partitions]
    if 'test_stripped' in partitions:
        assert len(partitions) == 1

    # Get image files and labels
    images, category_ids = [], []    
    for partition in partitions:
        coco = COCO('{}/annotations/{}_{}.json'.format(data_root, dataset, partition))
        imgIds = coco.getImgIds()
        images += [('{}/{}'.format(data_root, img['file_name']), img['id']) for img in coco.loadImgs(imgIds)]
        if partition != 'test_stripped':
            category_ids += [int(ann['category_id']) for ann in coco.loadAnns(coco.getAnnIds(imgIds=imgIds))]
    if partitions[0] != 'test_stripped':
        labels = [cat - CATEGORY_ID_BASE[dataset] - 1 for cat in category_ids]
    else:
        labels = None

    # Load normalization constants
    with open(data_root + '/decathlon_mean_std.pickle', 'rb') as handle:
        try:
            dict_mean_std = pickle.load(handle)
        except Exception:
            dict_mean_std = pickle.load(handle, encoding='bytes')
    means = dict_mean_std[(dataset + 'mean').encode('UTF-8')]
    stds = dict_mean_std[(dataset + 'std').encode('UTF-8')]

    # Transformations
    if partitions[0] == 'train':
        if dataset in ['svhn', 'omniglot']: # no horz flip 
            transform = transforms.Compose([
                transforms.Resize(72),
                transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        elif dataset in ['aircraft', 'daimlerpedcls', 'cifar100']:
            transform = transforms.Compose([
                transforms.Resize((72, 72)),
                transforms.Pad(8, padding_mode='reflect'),
                transforms.RandomCrop(72),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        elif dataset in ['gtsrb']: # Color jitter
            transform = transforms.Compose([
                transforms.Resize((72, 72)),
                transforms.Pad(8, padding_mode='reflect'),
                transforms.RandomCrop(72),
                transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        elif dataset in ['dtd']:
            transform = transforms.Compose([
                transforms.Resize(72),
                transforms.RandomCrop(72),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(72),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])  
    else:
        if dataset in ['omniglot', 'svhn']: # no horz flip 
            transform = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        elif dataset in ['aircraft', 'daimlerpedcls', 'cifar100', 'gtsrb']:
            transform = transforms.Compose([
                transforms.Resize((72, 72)),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        elif dataset in ['dtd']:
            transform = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

    # Dataset
    return DecathlonImageFolder(data_root, imgs=images, labels=labels, transform=transform, dataset=dataset, classes=coco.cats)


def get_dataloader(dataset, batch_size=1, shuffle=True, mode='train', num_workers=4):
    if dataset.startswith('decathlon'):
        partitions = ['train', 'val'] if mode == 'train' else ['test']
        dataset = get_decathlon_dataset(dataset.split('/')[1], partitions)
    else:
        dataset = get_dataset(dataset, mode)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


if __name__ == '__main__':
    import time
    import sys
    import numpy as np

    loader = get_dataloader(dataset='svhn', mode='train', batch_size=64, num_workers=2)
    print(loader.dataset)

    img, lbl, fn = loader.dataset[0]
    print(np.array(img).shape)