import torch
import numpy as np


def psnr(img_1, img_2):
    max_value = 1.0

    assert type(img_1) == type(img_2)

    if isinstance(img_1, np.ndarray):
        mse = np.mean((img_1 - img_2) ** 2)
    elif isinstance(img_1, torch.Tensor):
        mse = torch.mean((img_1 - img_2) ** 2).item()
    else:
        raise ValueError(f"Unrecognised data type: {type(img_1)} is neither numpy array or torch tensor")
    if mse == 0:
        psnr_ = 100
    else:
        psnr_ = 10 * np.log10(max_value ** 2 / mse)
    return psnr_



def l2_norm(x):
    if isinstance(x, np.ndarray):
        out = np.sum(x ** 2)
    else:
        out = torch.sum(x ** 2)
        if isinstance(out, torch.Tensor):
            out = out.item()
    return out


def relative_error(x, y_ref):
    return l2_norm(x - y_ref) / l2_norm(y_ref)
