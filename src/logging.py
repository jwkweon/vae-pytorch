import wandb as wb


def init_wandb(args):
    name = "-".join([args.model, args.dataset, str(args.learning_rate)])
    wb.init(
        project=args.project,
        name=name,
        settings=wb.Settings(code_dir=args.log_dir),
    )
    wb.config.update(args)
    return wb


def log_loss(loss, recon, kld, idx):
    wb.log(
        {
            "Total_loss": loss,
            "recon_loss": recon,
            "kld_loss": kld,
            "step": idx,
        }
    )
