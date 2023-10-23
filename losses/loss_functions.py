"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th
import torch.nn.functional as F

from common.tensor_dict import TensorDict


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    res = 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))
    return res


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def mse(pred, target, mean=True):
    if mean:
        loss = F.mse_loss(target, pred)
    else:
        loss = F.mse_loss(target, pred, reduction='none')
    return loss


def mse_loss_from_weights_dict(target_weights_dict: TensorDict, pred_weights_dict: TensorDict):
    loss = 0
    loss_info = {}
    for key in pred_weights_dict.keys():
        key_loss = F.mse_loss(target_weights_dict[key], pred_weights_dict[key])
        loss += key_loss
        loss_info[key] = key_loss.item()

    with th.no_grad():
        obsnorm_loss = th.Tensor([loss_info[key] for key in loss_info.keys() if 'obs_normalizer' in key]).sum()
        loss_info.update({
            'mse_loss': (loss - obsnorm_loss).item(),
            'obsnorm_loss': obsnorm_loss.item()
        })
    return loss, loss_info


def mse_from_norm_dict(target_weights_dict: dict, rec_weight_dict: dict):
    loss = 0
    loss_info = {}
    for key in rec_weight_dict.keys():
        key_loss = F.mse_loss(target_weights_dict[key], rec_weight_dict[key])
        loss += key_loss
        loss_info[key] = key_loss.item()
    return loss, loss_info
