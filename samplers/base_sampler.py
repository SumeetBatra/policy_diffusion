import torch
import torch.nn as nn
import numpy as np
import random

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Mapping, Callable, Dict
from common.utils import extract


class BaseSampler(ABC):
    determinstic: bool = False
    random_seed: int = 1
    scaler: torch.cuda.amp.GradScaler

    # redundant params from diffusion class but are needed here for reverse diffusion process
    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor

    def __init__(self, kwargs: Mapping[str, Any]):
        # TODO: need to somehow enforce that all the params above are passed in and set
        BaseSampler.set_attributes(self, kwargs)

        if self.random_seed != 0:
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        torch.backends.cudnn.benchmark = not self.determinstic
        torch.backends.cudnn.deterministic = self.determinstic

        self.scaler = torch.cuda.amp.GradScaler(**kwargs.pop("amp", {"enabled": False}))

    @classmethod
    def set_attributes(cls, obj: Any, values: Mapping[str, Any]) -> None:
        """Uses annotations to set the attributes of the instance object."""
        ann = vars(cls).get("__annotations__")
        if not isinstance(ann, dict):
            return
        for name in ann.keys():
            if (value := values.pop(name, None)) is not None:
                setattr(obj, name, value)

    @abstractmethod
    def sample(self,
               model: nn.Module,
               shape: List[int],
               cond: Optional[Dict]):
        """
        Iteratively performs the full reverse diffusion process
        :param model: NN model that learns the denoising
        :param shape: shape of the batch of noise we are denoising
        :param cond: condition if conditional generation
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def p_sample(self,
                 model: nn.Module,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 c: Optional[Dict] = None):
        """
        Perform one step of the reverse diffusion process
        :param model: NN model
        :param x: batch of samples to denoise
        :param c: condition (for conditional generation)
        :param t: timestep(s) we are at in the reverse process
        :return: one-step denoised x, predicted x0, noise at time t e_t
        """
        raise NotImplementedError

    def q_posterior_mean_variance(self, x_start, x_t, t):
        '''
        Compute the mean and variance of the diffusion posterior:
                q(x_{t-1} | x_t, x_0)
        '''
        assert x_start.shape == x_t.shape
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
                         extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        assert (
                posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_xstart_from_eps(self,
                                x_t: torch.Tensor,
                                t: torch.Tensor,
                                eps: torch.Tensor):
        """
        Predict x0 given the predicted noise
        :param x_t: batch of samples
        :param t: timestep(s) of the reverse diffusion process we are at
        :param eps: predicted noise
        :return:
        """
        assert x_t.shape == eps.shape
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps

