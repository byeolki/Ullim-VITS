import torch


def kl_divergence_loss(z_q, m_q, logs_q, m_p, logs_p, mask):
    logs_p = torch.clamp(logs_p, min=-10, max=10)
    logs_q = torch.clamp(logs_q, min=-10, max=10)

    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_q - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = kl - 0.5 * torch.exp(2.0 * (logs_q - logs_p))
    kl = torch.sum(kl * mask)
    loss = kl / torch.sum(mask)

    loss = torch.clamp(loss, min=0.0, max=1000.0)

    return loss


def duration_loss(logw, logw_, phoneme_lengths):
    if logw.dim() == 2:
        logw = logw.unsqueeze(1)
    if logw_.dim() == 2:
        logw_ = logw_.unsqueeze(1)

    logw = torch.clamp(logw, min=-10, max=10)
    logw_ = torch.clamp(logw_, min=-10, max=10)

    batch_size = logw.size(0)
    max_len = logw.size(2)
    mask = torch.arange(max_len, device=logw.device).unsqueeze(0) < phoneme_lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).float()

    loss = torch.sum((logw - logw_) ** 2 * mask) / torch.sum(mask)

    return loss
