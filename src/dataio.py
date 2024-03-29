import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, MNIST, CelebA


class Dataset_loader:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.batch_size = args.batch_size

    def load_data(self):
        if self.dataset == "mnist":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            train_dataset = datasets.MNIST(
                root="./datasets", train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root="./datasets", train=False, download=True, transform=transform
            )

        elif self.dataset == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    # ),
                ]
            )
            train_dataset = datasets.CIFAR10(
                root="./datasets", train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root="./datasets", train=False, download=True, transform=transform
            )

        elif self.dataset == "celeba":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    transforms.Resize((64, 64)),
                ]
            )
            # This code doesn't work
            # please download the dataset manually

            # train_dataset = datasets.CelebA(
            #     root="./datasets", split="train", download=True, transform=transform
            # )
            # test_dataset = datasets.CelebA(
            #     root="./datasets", split="test", download=True, transform=transform
            # )

            datapath = "./datasets/celeba/"
            train_dataset = datasets.ImageFolder(root=datapath, transform=transform)
            test_dataset = None

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if test_dataset is None:
            return train_loader, None
        else:
            test_loader = DataLoader(
                dataset=test_dataset, batch_size=self.batch_size, shuffle=False
            )
            return train_loader, test_loader
