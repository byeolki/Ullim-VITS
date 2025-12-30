import torch.optim as optim


def get_optimizer(model, discriminator, config):
    gen_params = list(model.parameters())

    optimizer_g = optim.AdamW(
        gen_params,
        lr=config.train.optimizer.generator.lr,
        betas=config.train.optimizer.generator.betas,
        eps=config.train.optimizer.generator.eps
    )

    disc_params = list(discriminator.parameters())

    optimizer_d = optim.AdamW(
        disc_params,
        lr=config.train.optimizer.discriminator.lr,
        betas=config.train.optimizer.discriminator.betas,
        eps=config.train.optimizer.discriminator.eps
    )

    return optimizer_g, optimizer_d
