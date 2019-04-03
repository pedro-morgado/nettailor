import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
from pycocotools.coco import COCO
import pickle
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def pil_loader(path):
    return Image.open(path).convert('RGB')

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

class ImageFolder(data.Dataset):
    def __init__(self, root, imgs=None, labels=None, transform=None, dataset=None, classes=None):
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.num_classes = NUM_CLASSES[dataset]
        self.category_ids = list(range(CATEGORY_ID_BASE[dataset]+1, CATEGORY_ID_BASE[dataset]+NUM_CLASSES[dataset]+1))
        self.classes = classes

    def shuffle(self):
        idx = range(len(self))
        random.shuffle(idx)
        self.imgs = [self.imgs[i] for i in idx]
        if self.labels is not None:
            self.labels = [self.labels[i] for i in idx]

    def __getitem__(self, index):
        img_id = self.imgs[index][1]
        img = pil_loader(self.imgs[index][0])
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index] if self.labels is not None else 0
        return img, target, img_id

    def __len__(self):
        return len(self.imgs)


def get_dataset(dataset, partitions):
    data_root = 'data/decathlon-1.0'
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
    return ImageFolder(data_root, imgs=images, labels=labels, transform=transform, dataset=dataset, classes=coco.cats)


def get_dataloader(dataset, partitions, batch_size=128, shuffle=False, num_workers=4):
    dataset = get_dataset(dataset, partitions)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)