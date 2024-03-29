from typing import List

import torch
import yaml
from einops import rearrange


def load_config(cofig_path):
    with open(cofig_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def torch2np_clip(tensor):
    tensor = rearrange(tensor, "b c h w -> b h w c")
    nptensor = tensor.detach().cpu().numpy()

    return np.clip(nptensor, 0, 1)


def viz_ori_and_recon(x, x_hat, args, idx):
    from matplotlib import pyplot as plt

    x = torch2np(x)
    x_hat = torch2np(x_hat)

    fig, axs = plt.subplots(5, 10, figsize=(20, 10))

    for i in range(5):
        for j in range(5):
            axs[i, j * 2].imshow(x[i * 5 + j])
            axs[i, j * 2].axis("off")  # 축 제거

            axs[i, j * 2 + 1].imshow(x_hat[i * 5 + j])
            axs[i, j * 2 + 1].axis("off")  # 축 제거

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(f"./outputs/{args.dataset}_result_{idx}.png")


def check_reparameterization(rep_z: List, z: List) -> None:
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    axs[0].hist(rep_z, bins=500, color="dodgerblue", alpha=0.6)
    axs[0].set_xlabel("Reparameterized Z")

    axs[1].hist(z, bins=500, color="crimson", alpha=0.6)
    axs[1].set_xlabel("Z~N(5, 3)")

    axs[2].hist(rep_z, bins=500, color="dodgerblue", alpha=0.3)
    axs[2].hist(z, bins=500, color="crimson", alpha=0.3)
    axs[2].set_xlabel("Compare two distributions")

    plt.show()


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)  # logvar = log(std^2)
    eps = torch.randn_like(std)

    return mu + eps * std


if __name__ == "__main__":
    num_points = 10000

    stdnorm_mu = torch.tensor([5.0] * num_points)
    stdnorm_var = torch.tensor([3.0] * num_points)

    rep_z = reparameterization(stdnorm_mu, stdnorm_var)
    z = torch.normal(5, 3**0.5, size=(1, num_points))

    check_reparameterization(rep_z, z)
