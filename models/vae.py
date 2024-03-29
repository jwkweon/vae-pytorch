from typing import List

import torch
import torch.nn as nn
from src.utils import reparameterization


class VAE(nn.Module):
    def __init__(self, hidden_dims: List[int]):
        super(VAE, self).__init__()
        self.hidden_dims = hidden_dims

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(
        self,
    ):
        enc_module = []
        for i in range(len(self.hidden_dims) - 2):
            enc_module.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            enc_module.append(nn.ReLU())

        encoder = nn.Sequential(*enc_module)
        self.mu_linear = nn.Sequential(
            nn.Linear(self.hidden_dims[-2], self.hidden_dims[-1])
        )
        self.var_linear = nn.Sequential(
            nn.Linear(self.hidden_dims[-2], self.hidden_dims[-1])
        )

        return encoder

    def build_decoder(
        self,
    ):
        dec_module = []

        for i in range(len(self.hidden_dims) - 1, 0, -1):
            if i == len(self.hidden_dims) - 1:
                dec_module.append(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1])
                )
            else:
                dec_module.append(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i - 1])
                )

            if i != 1:
                dec_module.append(nn.ReLU())
            elif i == 1:
                dec_module.append(nn.Sigmoid())

        decoder = nn.Sequential(*dec_module)
        return decoder

    def forward(self, x):
        _, c, h, w = x.size()

        x = x.view(-1, self.hidden_dims[0])
        x = self.encoder(x)

        # mu, var = torch.chunk(x, 2, dim=-1)
        mu = self.mu_linear(x)
        logvar = self.var_linear(x)
        z = reparameterization(mu, logvar)

        x = self.decoder(z)
        x = x.view(-1, c, h, w)

        return x, mu, logvar
