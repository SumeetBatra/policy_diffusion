import torch
import torch.nn as nn
import numpy as np

from typing import Optional, List, Any, Mapping
from samplers.base_sampler import BaseSampler


class DDIMSampler(BaseSampler):
    """
    Denoising Diffusion Implicit Models as described in the paper https://arxiv.org/abs/2010.02502
    This class implements a few other neat tricks:
        - Classifier free guidance: https://arxiv.org/abs/2207.12598
        - Composable Diffusion Models: https://arxiv.org/abs/2206.01714
    """
    def __init__(self,
                 n_steps: int,
                 ddim_discretize: str = "uniform",
                 ddim_eta: float = 0.,
                 **kwargs: Mapping[str, Any]):
        super().__init__(kwargs)
        BaseSampler.set_attributes(self, kwargs)

        self.n_steps = n_steps

        if ddim_discretize == 'uniform':
            # subsample every T/S steps, where T is the num_timesteps used during training, and S is the num
            # steps to take when sampling, where S < T
            c = self.n_steps // n_steps
            self.timesteps = np.asarray(list(range(0, self.n_steps, c))) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        # ddim sampling params
        self.ddim_alpha = self.alphas_cumprod[self.timesteps]
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_alpha_prev = torch.cat([self.alphas_cumprod[0:1], self.alphas_cumprod[self.timesteps[:-1]]])
        self.ddim_sigma = (ddim_eta *
                           ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                            (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** 0.5
                           )
        self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** 0.5

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               shape: List[int],
               cond: Optional[torch.Tensor] = None,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               classifier_free_guidance: bool = False,
               classifier_scale: int = 1.0):
        device = model.device
        bs = shape[0]
        x = x_last if x_last is not None else torch.randn(shape, device=device)
        timesteps = np.flip(self.timesteps)[skip_steps:]

        for i, step in enumerate(timesteps):
            index = len(timesteps) - i - 1
            ts = x.new_full(size=(bs,), fill_value=step, dtype=torch.long)
            x, pred_x0, e_t = self.p_sample(model, x, ts, cond, step, index=index, repeat_noise=repeat_noise,
                                            temperature=temperature, classifier_free_guidance=classifier_free_guidance,
                                            classifier_scale=classifier_scale)
        return x

    @torch.no_grad()
    def p_sample(self,
                 model: nn.Module,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 c: Optional[torch.Tensor],
                 step: int,
                 index: int,
                 *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 classifier_free_guidance: bool = False,
                 classifier_scale: int = 1.0):
        e_t = None
        if classifier_free_guidance:
            e_t_uncond = model(x, t, cond=None)
            if len(c.shape) == 2:
                # standard classifier free guidance
                e_t = model(x, t, c)
                e_t = (1.0 + classifier_scale) * e_t - classifier_scale * e_t_uncond
            elif len(c.shape) == 3:
                dim0, dim1 = c.shape[0], c.shape[1]
                x_orig_shape = x.shape

                c = c.reshape(dim0 * dim1, -1)
                x_cpy = torch.repeat_interleave(x, repeats=dim1, dim=0)
                e_t = model(x_cpy, t, c)
                e_t = e_t_uncond + ((classifier_scale + 1.0) * (e_t - e_t_uncond)).view((dim0, dim1,) + x_orig_shape[1:]).sum(dim=1)
            else:
                raise NotImplementedError(f'Unsupported condition dimensionality {c.shape} passed in. Must be 2 or 3 dim')
        else:
            e_t = model(x, t, c)

        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x, temperature=temperature, repeat_noise=repeat_noise)
        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(self,
                               e_t: torch.Tensor,
                               index: int,
                               x: torch.Tensor,
                               temperature: float,
                               repeat_noise: bool):
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)

        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        if sigma == 0.:
            noise = 0.
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise *= temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0