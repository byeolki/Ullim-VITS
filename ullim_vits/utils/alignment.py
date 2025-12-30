import torch
import numpy as np


def maximum_path(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)

    for i in range(neg_cent.shape[0]):
        path[i] = maximum_path_c(neg_cent[i], t_t_max[i], t_s_max[i])

    return torch.from_numpy(path).to(device=device, dtype=dtype)


def maximum_path_c(neg_cent, t_t_max, t_s_max):
    path = np.zeros((t_t_max, t_s_max), dtype=np.int32)

    t_s = t_s_max
    for t_t in range(t_t_max - 1, -1, -1):
        max_idx = 0
        max_val = neg_cent[t_t, 0]

        for t_s_candidate in range(1, min(t_s + 1, t_s_max)):
            if neg_cent[t_t, t_s_candidate] > max_val:
                max_val = neg_cent[t_t, t_s_candidate]
                max_idx = t_s_candidate

        path[t_t, max_idx] = 1
        t_s = max_idx

    return path


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, dim=1)

    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype, device=device)

    for i in range(b):
        cum_dur = cum_duration[i]
        for j in range(t_x):
            start = 0 if j == 0 else cum_dur[j - 1]
            end = cum_dur[j]
            path[i, j, start:end] = 1

    return path * mask
