import torch


def feature_matching_loss(fmap_r, fmap_g):
    loss = 0
    num_features = 0

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
            num_features += 1

    return loss / num_features if num_features > 0 else loss
