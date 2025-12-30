import torch.optim as optim


def get_scheduler(optimizer, config):
    if config.train.scheduler.type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.train.scheduler.gamma
        )
    elif config.train.scheduler.type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs
        )
    else:
        scheduler = None

    return scheduler
