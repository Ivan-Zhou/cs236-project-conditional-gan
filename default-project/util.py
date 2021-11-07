import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, imsize, batch_size, eval_size, num_workers=1):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    def target_to_oh(target):
        NUM_CLASS = 120  # hard code here....
        one_hot = torch.eye(NUM_CLASS)[target]
        return one_hot

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
        target_transform=target_to_oh,
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


def get_dataloaders(data_dir, imsize, batch_size, eval_size, num_workers=1, dataset="local"):
    if dataset == "local":
        return get_dataloaders_from_local(data_dir, imsize, batch_size, eval_size, num_workers=1)
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
