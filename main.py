import argparse
import os.path as osp

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.vae import VAE
from src.dataio import Dataset_loader
from src.logging import init_wandb
from src.loss import loss_fn
from src.optimizer import get_optimizer
from src.utils import load_config
from trainer import Trainer

# TODO
# [] Type hinting
# [] Add HVAE
# [] Add MHVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--config", type=str, default="vae.yaml")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(osp.join("./config", args.config))
    for key, value in config.items():
        setattr(args, key, value)

    if args.dataset == "mnist":
        args.hidden_dims = [28 * 28, 512, 10]

    elif args.dataset == "cifar10":
        args.hidden_dims = [32 * 32 * 3, 512, 256, 256, 10]

    elif args.dataset == "celeba":
        args.batch_size = 64
        args.hidden_dims = [64 * 64 * 3, 1024, 512, 256, 10]

    train_loader, test_loader = Dataset_loader(args).load_data()
    model = VAE(args.hidden_dims).to(args.device)
    args.optimizer = get_optimizer(model, args)
    args.loss_fn = loss_fn

    print(model)
    print(len(train_loader))

    init_wandb(args)
    trainer = Trainer(args, model, train_loader, test_loader)
    trainer.train()
