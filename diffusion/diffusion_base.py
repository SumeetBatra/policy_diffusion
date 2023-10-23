import torch.nn.functional as F
import logging

from trainers.trainer_base import TrainerBase
from diffusion.schedules import *
from typing import Mapping, Any, Optional
from common.utils import extract, create_instance_from_spec as from_spec
from samplers.base_sampler import BaseSampler
from torch.utils.tensorboard import SummaryWriter

BETA_SCHEDULE_REGISTRY = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
    "quadratic": quadratic_beta_schedule,
    "sigmoid": sigmoid_beta_schedule
}

logger = logging.getLogger("diffusion")
logger.setLevel(logging.DEBUG)


class DiffusionBase(TrainerBase):
    sampler: BaseSampler

    # empirically determined factor by which we scale the generated latent codes
    scale_factor: float

    # discrete time diffusion default attributes
    timesteps: int = 1000
    beta_schedule: str = "linear"
    betas: torch.Tensor = BETA_SCHEDULE_REGISTRY[beta_schedule](timesteps)

    # define alphas
    alphas: torch.Tensor = 1. - betas
    alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev: torch.Tensor = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas: torch.Tensor = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    # we can compute this directly for any intermediate timestep b/c sum of gaussians is gaussian,
    # giving us a closed form solution
    posterior_variance: torch.Tensor = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    # variance is 0 at beginning of diffusion chain so we "clip" it by replacing the 0th index with the 1st
    posterior_log_variance_clipped: torch.Tensor = torch.log(torch.cat((posterior_variance[1].view(-1, 1), posterior_variance[1:].view(-1, 1))).squeeze())

    # equation 11 first term in improved DDPM
    posterior_mean_coef1: torch.Tensor = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # equation 11 second term in Improved DDPMs
    posterior_mean_coef2: torch.Tensor = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    sqrt_recip_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 / alphas_cumprod - 1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, spec: Mapping[str, Any]) -> None:
        self.spec = spec
        self.initialize_datasets()
        self.initialize_env()

        logvar = torch.full(fill_value=0., size=(self.timesteps,))
        self.model = from_spec(spec['model'], logvar=logvar)

        self.sampler = from_spec(spec['sampler'])

        if self.use_wandb:
            self.writer = SummaryWriter(f'runs/{self.name}')

        self.optimizer = from_spec(spec['optim'], self.model.parameters())

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion process (using the "nice" property)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


