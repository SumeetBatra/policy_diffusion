import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import random
import logging

from schedules import *
from typing import Mapping, Any, Optional
from pathlib import Path
from dataset.shaped_elites_dataset import shaped_elites_dataset_factory
from envs.brax_custom.brax_env import make_vec_env_brax
from common.brax_utils import shared_params
from common.utils import extract, create_instance_from_spec as from_spec
from samplers.base_sampler import BaseSampler
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter

BETA_SCHEDULE_REGISTRY = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
    "quadratic": quadratic_beta_schedule,
    "sigmoid": sigmoid_beta_schedule
}

logger = logging.getLogger("diffusion")
logger.setLevel(logging.DEBUG)


class DiffusionBase:
    name: str

    random_seed: int = 1

    env_name: str
    exp_dir: str

    save_image_path: str

    checkpoint_n_epochs: int

    deterministic: bool = False

    scaler: torch.cuda.amp.GradScaler

    track_agent_quality: bool = False

    use_language: bool = False

    # log results to wandb
    use_wandb: bool = False

    # debugging mode enables logging of extram params like grad norm
    debug: bool = False

    # log results to tensorboard
    writer: SummaryWriter

    # whether to re-evaluate the archive using the model. Used to track training performance
    reeval_archive: bool = False
    # whether to perform the cut out experiment
    cut_out: bool = False
    # whether to average elites when reevaluating the archive
    average_elites: bool = False
    # whether to center the data. This is done by dividing each layer of each policy by the mean and std-dev
    # over all policies for that layer
    center_data: bool = False

    grad_clip: bool = False

    train_batch_size: int = 1
    test_batch_size: int = 1

    start_epoch: int = 1
    num_epochs: int = 1
    global_step: int = 1

    model: nn.Module
    optimizer: torch.optim.Optimizer

    sampler: BaseSampler

    device: str

    # latent space params for the VAE. In the quantization layer, z -> latent (emb) -> z
    latent_channels: int
    latent_size: int
    z_channels: int

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
        DiffusionBase.set_attributes(self, kwargs)

        self.spec = None
        self.train_loader = None
        self.train_archive = None
        self.weight_normalizer = None
        self.test_loader = None

        self.cp_dir = self.exp_dir.joinpath('checkpoints')
        self.cp_dir.mkdir(exist_ok=True)

        # env specific params
        self.obs_dim = None
        self.action_shape = None
        self.env = None
        self.clip_obs_rew = None

        self.sampler = None

        if self.random_seed != 0:
            torch.manual_seed(self.random_seed)
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        torch.backends.cudnn.benchmark = not self.determinstic
        torch.backends.cudnn.deterministic = self.determinstic

        self.scaler = torch.cuda.amp.GradScaler(**kwargs.pop("amp", {"enabled": False}))

    def initialize_datasets(self) -> None:
        self.train_loader, self.train_archive, self.weight_normalizer = shaped_elites_dataset_factory(
            env_name=self.env_name,
            batch_size=self.train_batch_size,
            is_eval=False,
            center_data=self.center_data,
            results_folder=self.exp_dir,
            use_language=self.use_language,
            weight_normalizer=self.weight_normalizer
        )
        self.test_loader, *_ = shaped_elites_dataset_factory(
            env_name=self.env_name,
            batch_size=self.test_batch_size,
            is_eval=True,
            center_data=self.center_data,
            results_folder=self.exp_dir,
            use_language=self.use_language,
            weight_normalizer=self.weight_normalizer
        )

    def initialize_env(self, spec: Mapping[str, Any]) -> None:
        self.env_name = spec['env']['env_name']
        self.obs_dim, self.action_shape = shared_params[self.env_name], np.array([shared_params[self.env_name]['action_dim']])
        self.clip_obs_rew = spec['env']['clip_obs_rew']

        if self.track_agent_quality:
            self.env = make_vec_env_brax(spec['env'])

    def build(self, spec: Mapping[str, Any]) -> None:
        self.spec = spec
        self.initialize_datasets(spec)
        self.initialize_env(spec)

        logvar = torch.full(fill_value=0., size=(self.timesteps,))
        self.model = from_spec(spec['model'], logvar=logvar)

        self.sampler = from_spec(spec['sampler'])

        if self.use_wandb:
            self.writer = SummaryWriter(f'runs/{self.exp_name}')

        self.optimizer = from_spec(spec['optim'], self.model.parameters())

    @classmethod
    def set_attributes(cls, obj: Any, values: Mapping[str, Any]) -> None:
        """Uses annotations to set the attributes of the instance object."""
        ann = vars(cls).get("__annotations__")
        if not isinstance(ann, dict):
            return
        for name in ann.keys():
            if (value := values.pop(name, None)) is not None:
                setattr(obj, name, value)

    def save_checkpoint(self, epoch: int, global_step: int):
        suffix = f'epoch_{epoch}_iteration_{global_step}'
        filename = self.cp_dir.joinpath(f'diffusion_model_checkpoint_{suffix}.pt')
        logger.debug(f'Saving {self.exp_name} to {filename}')

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "weight_normalizer": self.weight_normalizer.state_dict()
            },
            filename
        )

    def load_checkpoint(self, path: str):
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Checkpoint {path} does not exist.")
        logger.info(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optim"])
        if (scaler_state := checkpoint.get("scaler")) is not None:
            self.scaler.load_state_dict(scaler_state)

        self.weight_normalizer.load_state_dict(checkpoint['weight_normalizer'])

        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"] + 1

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError

    @abstractmethod
    def compute_training_losses(self,
                                model: nn.Module,
                                x_start: torch.Tensor,
                                t: torch.Tensor,
                                model_kwargs: Optional[Mapping[Any, Any]] = None,
                                noise: Optional[torch.Tensor] = None):
        raise NotImplementedError

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


