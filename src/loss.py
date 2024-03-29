import torch


def loss_fn(x, x_hat, mu, logvar):
    # reconstruction term
    # E[log P(x|z)]
    # x_hat : generated image, x : original image
    recon_err = torch.nn.functional.mse_loss(x_hat, x, reduction="sum")
    # recon_err = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    # prior matching term
    # KLD : (q(z|x) || p(z))
    # kld = 0.5 * torch.sum(-1 - torch.exp(logvar) + mu**2 + logvar)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # total loss : Recon + KLD
    loss = recon_err + kld

    return loss, recon_err, kld
