import torch
import torch.nn as nn

from samplers.base_sampler import BaseSampler
from typing import Optional, List, Callable, Any, Mapping
from tqdm import tqdm
from common.utils import extract


class DDPMSampler(BaseSampler):
    """
    The standard sampler used in the original DDPM paper
    """
    def __init__(self, **kwargs: Mapping[str, Any]):
        super().__init__(kwargs)
        BaseSampler.set_attributes(self, kwargs)

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               shape: List[int],
               cond: Optional[torch.Tensor]):
        device = next(model.parameters()).device

        # start from pure noise (for each sample in the batch)
        samples = torch.randn(shape, device=device)
        bs = shape[0]

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            samples = self.p_sample(model, samples, torch.full((bs,), i, device=device, dtype=torch.long), cond)['sample']

        return samples

    @torch.no_grad()
    def p_sample(self,
                 model: nn.Module,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 c: Optional[torch.Tensor] = None):
        out = self.p_mean_variance(model, x, t, c, clip_denoised=True, denoised_fn=None)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(self,
                        model: nn.Module,
                        x: torch.Tensor,
                        t: torch.Tensor,
                        c: Optional[torch.Tensor] = None,
                        clip_denoised: bool = True,
                        denoised_fn: Callable = None):
        '''
        Apply the model to get p(x_{t-1} | x_t) as well as prediction of initial x_0
        As in the improved DDPM paper, the model jointly predicts the mean and variance
        '''
        # TODO: redundant with method in diffusion class. Figure out best place to put this only once
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, c)

        # assume that we learn the variance within a range given by B_t and Bhat_t (posterior variance)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = torch.split(model_output, C, dim=1)

        # learn the variance between a range
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        # the model_var_values is [-1, 1] for [min_var, max_var]. Need to shift it to [0, 1] range
        frac = (model_var_values + 1) / 2
        # equation 15 in Improved DDPM
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # assume that we predict epsilon noise
        pred_xstart = process_xstart(self.predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart
        }