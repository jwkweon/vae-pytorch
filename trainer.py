import torch
from tqdm.auto import trange

from src.logging import log_loss
from src.utils import viz_ori_and_recon

# TODO
# [] Implement the Trainer class
# [v] Define loss
# [v] set args.device
# [v] set args.optimizer
# [] save models
# [v] save logs


class Trainer:
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = args.optimizer

    def iter_dataset(self, data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(self.args.device), y.to(self.args.device)

    def train(self):
        for idx in (pbar := trange(self.args.train_iter)):
            x, _ = next(self.iter_dataset(self.train_loader))

            self.optimizer.zero_grad()
            x_hat, mu, logvar = self.model(x)

            loss, recon, kld = self.args.loss_fn(x, x_hat, mu, logvar)
            loss.backward()
            self.optimizer.step()
            pbar.set_description(
                f"Loss: {loss.item():.4f} | Recon: {recon.item():.4f} | KLD: {kld.item():.4f}"
            )

            if idx % self.args.log_iter == 0:
                log_loss(loss, recon, kld, idx)

            if idx % self.args.log_fig_iter == 0 and self.args.save_fig:
                viz_ori_and_recon(x, x_hat, self.args, idx)
