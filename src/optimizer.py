import torch.optim as optim


# TODO
# 1. Add arguments detail, like beta1, beta2, momentum, etc.
def get_optimizer(model, args):
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Invalid optimizer")
    return optimizer
