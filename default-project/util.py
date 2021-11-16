import glob
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_dataloaders_from_local(data_dir, imsize, batch_size, eval_size, num_workers=1):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    # def target_to_oh(target):
    #     num_classes=120 #TODO(yuanzhe): no hardcode...
    #     one_hot = torch.eye(num_classes)[target]
    #     return one_hot

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        #target_transform=target_to_oh,
    )
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}

    eval_dataset, train_dataset = torch.utils.data.random_split(
        dataset,
        [eval_size, len(dataset) - eval_size],
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_dataloader, eval_dataloader, dataset.class_to_idx, idx2class


def get_dataloaders(data_dir, imsize, batch_size, eval_size, num_workers=1, dataset="mnist"):
    if dataset in ["stanford_dog", "stanford_dogs_top_10"]:  # dataset from local directory
        assert os.path.exists(data_dir), f"The directory {data_dir} does not exist!"
        return get_dataloaders_from_local(data_dir, imsize, batch_size, eval_size, num_workers=num_workers)
    else:
        assert dataset in ["mnist", "fashion-mnist", "cifar10", "svhn", "stl10"], print(f"Unsupported dataset {dataset}")
        train_dataloader = dataloader(dataset, imsize, batch_size, split='train')
        eval_dataloader = dataloader(dataset, imsize, batch_size, split='eval')
        return train_dataloader, eval_dataloader, None, None
        

def dataloader(dataset, imsize, batch_size, split='train'):
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.Grayscale(3),
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if dataset == 'mnist':
        is_train = split == "train"
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=is_train, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        is_train = split == "train"
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=is_train, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        is_train = split == "train"
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=is_train, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    return data_loader


def get_num_classes_by_dataset(dataset):
    if dataset == "stanford_dog":
        return 120
    elif dataset == "mnist":
        return 10
    elif dataset == "stanford_dogs_top_10":
        return 10
    else:
        raise ValueError(f"No num_classes defined for the dataset {dataset}")


def get_eval_checkpoint(ckpt_path, exp_dir):
    if ckpt_path != "" and os.path.exists(ckpt_path):
        return ckpt_path
    if exp_dir != "" and os.path.exists(exp_dir):
        ckpt_list = glob.glob(os.path.join(exp_dir, "ckpt", "*.pth"))
        assert len(ckpt_list) > 0, f"No checkpoint can be found at {exp_dir}!"
        return max(ckpt_list, key=os.path.getctime)
    raise FileNotFoundError(f"No checkpoint can be found at path {ckpt_path} or directory {exp_dir}!")
