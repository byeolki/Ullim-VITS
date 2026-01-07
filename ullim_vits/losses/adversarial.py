import torch
import torch.nn.functional as F


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = torch.clamp(dr, min=-10, max=10)
        dg = torch.clamp(dg, min=-10, max=10)
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        dg = torch.clamp(dg, min=-10, max=10)
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l.item())
        loss += l

    return loss, gen_losses


def hinge_discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(F.relu(1.0 - dr))
        g_loss = torch.mean(F.relu(1.0 + dg))
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def hinge_generator_loss(disc_outputs):
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        l = -torch.mean(dg)
        gen_losses.append(l.item())
        loss += l

    return loss, gen_losses
