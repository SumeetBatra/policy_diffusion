# common trainer for both VAE and diffusion
import torch
import torch.nn as nn
import logging
import random
import numpy as np
import os

from abc import abstractmethod
from typing import Any, Mapping, Optional
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from dataset.shaped_elites_dataset import shaped_elites_dataset_factory
from envs.brax_custom.brax_env import make_vec_env_brax
from common.brax_utils import shared_params
from common.utils import extract, create_instance_from_spec as from_spec

logger = logging.getLogger("trainer")
logger.setLevel(logging.DEBUG)


class TrainerBase:
    name: str

    random_seed: int

    env_name: str
    exp_dir: str

    # where the trained archive policy "datasets" are saved
    archive_dir: str

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

    device: str

    # latent space params for the VAE. In the quantization layer, z -> latent (emb) -> z
    latent_channels: int
    latent_size: int
    z_channels: int

    def __init__(self, **kwargs):
        TrainerBase.set_attributes(self, kwargs)

        self.spec = None
        self.train_loader = None
        self.train_archive = None
        self.weight_normalizer = None
        self.test_loader = None

        self.cp_dir = Path(os.path.join(self.exp_dir, 'checkpoints'))
        self.cp_dir.mkdir(exist_ok=True)

        # env specific params
        self.obs_dim = None
        self.action_shape = None
        self.env = None
        self.clip_obs_rew = None

        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        torch.backends.cudnn.benchmark = not self.deterministic
        torch.backends.cudnn.deterministic = self.deterministic

        self.scaler = torch.cuda.amp.GradScaler(**kwargs.pop("amp", {"enabled": False}))

    def initialize_datasets(self) -> None:
        self.train_loader, self.train_archive, self.weight_normalizer = shaped_elites_dataset_factory(
            env_name=self.env_name,
            archive_dir=self.archive_dir,
            batch_size=self.train_batch_size,
            is_eval=False,
            center_data=self.center_data,
            results_folder=self.exp_dir,
            use_language=self.use_language,
            weight_normalizer=self.weight_normalizer
        )
        self.test_loader, *_ = shaped_elites_dataset_factory(
            env_name=self.env_name,
            archive_dir=self.archive_dir,
            batch_size=self.test_batch_size,
            is_eval=True,
            center_data=self.center_data,
            results_folder=self.exp_dir,
            use_language=self.use_language,
            weight_normalizer=self.weight_normalizer
        )

    def initialize_env(self, spec: Mapping[str, Any]) -> None:
        self.env_name = spec['env']['env_name']
        self.obs_dim, self.action_shape = shared_params[self.env_name]['obs_dim'], np.array([shared_params[self.env_name]['action_dim']])
        self.clip_obs_rew = spec['env']['clip_obs_rew']

        if self.track_agent_quality:
            rollouts_per_agent = 10  # to align ourselves with baselines
            spec['env']['env_batch_size'] = self.test_batch_size * rollouts_per_agent
            self.env = make_vec_env_brax(spec['env'], seed=self.random_seed)

    def build(self, spec: Mapping[str, Any]) -> None:
        self.spec = spec
        self.initialize_env(spec)
        self.initialize_datasets()

        self.model = from_spec(spec['model'])
        self.model.to(self.device)

        if self.use_wandb:
            self.writer = SummaryWriter(f'runs/{self.name}')

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
        logger.debug(f'Saving {self.name} to {filename}')

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
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

