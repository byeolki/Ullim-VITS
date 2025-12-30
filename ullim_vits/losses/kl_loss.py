import torch


def kl_divergence_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)

    return l


def duration_loss(logw, logw_, phoneme_lengths):
    l = torch.sum((logw - logw_) ** 2, [1]) / torch.sum(phoneme_lengths)
    l = torch.mean(l)

    return l
