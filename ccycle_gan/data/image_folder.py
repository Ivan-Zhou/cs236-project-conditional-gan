"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]
LABELS_MAP = {
    "has_glass": 1,
    "no_glass": 0
}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root_dir, max_dataset_size=float("inf")):
    images, labels = [], []
    assert os.path.isdir(root_dir), '%s is not a valid directory' % root_dir

    label = os.listdir(root_dir)
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        
        if not os.path.isdir(root_dir):
            continue
        
        for fname in sorted(os.listdir(label_dir)):
            if is_image_file(fname):
                path = os.path.join(label_dir, fname)
                images.append(path)
                labels.append(LABELS_MAP[label])
    n = min(max_dataset_size, len(images))
    return images[:n], labels[:n]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs, labels = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        label = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, labels, path
        else:
            return img, labels

    def __len__(self):
        return len(self.imgs)
