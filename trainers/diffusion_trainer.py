import torch.jit
import torch.nn.functional as F
import torch.nn as nn
import wandb
import time
import copy
import logging
import glob
import yaml

from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from diffusion.diffusion_base import DiffusionBase
from common.metrics import evaluate_agent_quality
from common.analysis import evaluate_ldm_subsample
from common.utils import grad_norm, create_instance_from_spec as from_spec
from typing import Dict, Any, Optional
from losses.loss_functions import mse

logger = logging.getLogger("policy_diffusion")


class PolicyDiffusion(DiffusionBase):
    def __init__(self, **kwargs):
        '''
        Implements the main policy diffusion model described in the paper. uses VAE for latent
        diffusion and DDIM for fast sampling
        '''
        super().__init__(**kwargs)

        self.clip_denoised = False
        self.vlb_weights = self.betas ** 2 / (2 * self.posterior_variance * self.alphas * (1 - self.alphas_cumprod))
        self.vlb_weights[0] = self.vlb_weights[1]
        self.vlb_loss_coef = 1e-5

    def build(self, spec: Dict[str, Any]) -> None:
        self.spec = spec
        self.initialize_env()
        self.initialize_datasets()

        logvar = torch.full(fill_value=0., size=(self.timesteps,))
        self.model = from_spec(spec['model'], logvar=logvar)
        self.model.to(self.device)

        self.autoencoder = self._load_autoencoder_from_checkpoint(self.spec)

        self.latent_channels = self.autoencoder.emb_channels
        self.z_channels = self.autoencoder.z_channels
        self.latent_size = self.autoencoder.z_height

        # create sampler
        diffusion_params = {
            "timesteps": self.timesteps,
            "betas": self.betas,
            "alphas": self.alphas,
            "alphas_cumprod": self.alphas_cumprod,
            "alphas_cumprod_prev": self.alphas_cumprod_prev,
            "posterior_variance": self.posterior_variance,
            "posterior_log_variance_clipped": self.posterior_log_variance_clipped,
            "posterior_mean_coef1": self.posterior_mean_coef1,
            "posterior_mean_coef2": self.posterior_mean_coef2,
            "sqrt_recip_alphas_cumprod": self.sqrt_recip_alphas_cumprod,
            "sqrt_recipm1_alphas_cumprod": self.sqrt_recipm1_alphas_cumprod
        }

        self.sampler = from_spec(spec['sampler'], **diffusion_params)

        if self.use_wandb:
            self.writer = SummaryWriter(f'runs/{self.name}')

        self.optimizer = from_spec(spec['optim'], self.model.parameters())

    def _load_autoencoder_from_checkpoint(self, spec: Dict[str, Any]) -> nn.Module:
        '''
        Load autoencoder from checkpoint
        :param path: path to autoencoder run dir
        '''
        autoencoder_cp_dir = spec['trainer']['config']['autoencoder_cp_dir']

        path = Path(autoencoder_cp_dir)
        if not path.exists():
            raise RuntimeError(f'Checkpoint dir {path} does not exist')

        spec_path = path.joinpath('experiment_spec_final.yaml')
        with open(spec_path, 'r', encoding='utf-8') as src:
            spec = yaml.safe_load(src)

        autoencoder = from_spec(spec['model'], obs_shape=self.obs_dim, action_shape=self.action_shape)
        autoencoder.to(self.device)

        cp_dir = path.joinpath('checkpoints')
        latest_cp = sorted(glob.glob(str(cp_dir) + '/model*'))[-1]

        checkpoint = torch.load(latest_cp, map_location=self.device)
        autoencoder.load_state_dict(checkpoint['model'])
        return autoencoder

    def compute_training_losses(self,
                                model: nn.Module,
                                x_start: torch.Tensor,
                                t: torch.Tensor,
                                model_kwargs: Optional[Dict[Any, Any]] = None,
                                noise: Optional[torch.Tensor] = None):
        cond = None
        if model_kwargs is None:
            model_kwargs = {}
        else:
            cond = model_kwargs['cond']
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        model_output = model(x_t, t, cond)  # TODO: implement conditioning via model_kwargs
        target = noise
        with torch.no_grad():
            output_mean, output_var = model_output.mean(), model_output.var()

        #  See https://arxiv.org/pdf/2112.10752.pdf Section B on why we can simplify vlb loss like this
        vlb_loss = mse(model_output, target, mean=False).mean([1, 2, 3])
        vlb_weights = self.vlb_weights.to(self.device)
        vlb_loss = (vlb_weights[t] * vlb_loss).mean()

        # simple loss term
        logvar_t = model.logvar[t]
        simple_loss = mse(model_output, target, mean=False).mean([1, 2, 3])
        simple_loss = (simple_loss / torch.exp(logvar_t)) + logvar_t
        simple_loss = simple_loss.mean()

        loss = simple_loss + self.vlb_loss_coef * vlb_loss
        loss_dict = {
            f'losses/simple_loss': simple_loss.mean().item(),
            f'losses/vlb_loss': vlb_loss.mean().item(),
        }
        info_dict = {
            f'train/log_var': model.logvar.mean().item(),
            f'data/model_output_mean': output_mean.item(),
            f'data/model_output_var': output_var.item(),
        }
        return loss, loss_dict, info_dict

    def train(self):
        self.model.train()

        logger.info('Computing the scale factor...')
        gt_params_batch, _ = next(iter(self.train_loader))
        with torch.no_grad():
            self.autoencoder.eval()
            batch = self.autoencoder.encode(gt_params_batch).sample().detach()
            # rescale the embeddings to be unit variance
            std = batch.flatten().std()
            self.scale_factor = 1. / std
            self.spec['scale_factor'] = self.scale_factor

        logger.info('Starting training loop')
        start_time = time.time()
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # do an initial round of validation
            if self.track_agent_quality and epoch % 10 == 0:
                info = self.validate()
                image_results, uniform_image_results = None, None
                if epoch % 50 == 0 and self.reeval_archive:
                    info, image_results, uniform_image_results = self.reevaluate_archive(epoch, info)

                if self.use_wandb:
                    for key, val in info.items():
                        self.writer.add_scalar(key, val, self.global_step + 1)

                    info.update({
                        'global_step': self.global_step + 1,
                        'epoch': epoch + 1
                    })

                    wandb.log(info)
                    if self.reeval_archive and image_results is not None:
                        wandb.log({'Archive/recon_image': wandb.Image(image_results['Reconstructed'], caption=f"Epoch {epoch + 1}")})
                        wandb.log({'Archive/Uniform_recon_image': wandb.Image(uniform_image_results['Reconstructed'], caption=f"Epoch {epoch + 1}")})

            # now the main training loop begins
            epoch_simple_loss = 0
            epoch_vlb_loss = 0
            epoch_grad_norm = 0
            self.model.train()
            for step, (policies, measures) in enumerate(tqdm(self.train_loader)):
                if self.use_language:
                    measures, text_labels = measures
                self.optimizer.zero_grad()

                measures = measures.to(torch.float32).to(self.device)

                with torch.no_grad():
                    batch = self.autoencoder.encode(policies).sample().detach()
                    batch *= self.scale_factor

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (len(policies),), device=self.device).long()

                if self.use_language:
                    cond = self.model.text_to_cond(text_labels)
                else:
                    cond = measures

                losses, loss_dict, info_dict = self.compute_training_losses(self.model,
                                                                            batch,
                                                                            t,
                                                                            model_kwargs={'cond': cond})
                loss = losses.mean()
                loss.backward()

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.global_step += 1

                if self.debug:
                    epoch_grad_norm += grad_norm(self.model)

                epoch_simple_loss += loss_dict['losses/simple_loss']
                epoch_vlb_loss += loss_dict['losses/vlb_loss']

                # logging at the per-optimizer-step timescale
                if self.use_wandb:
                    wandb.log(info_dict)
                    wandb.log({
                        'data/batch_mean': batch.mean().item(),
                        'data/batch_var': batch.var().item(),
                        'global_step': self.global_step + 1
                    })

            # logging at the per-epoch timescale
            logger.info(f'Epoch: {epoch} Simple loss: {epoch_simple_loss / len(self.train_loader)}, Vlb Loss: {epoch_vlb_loss / len(self.train_loader)}')
            if self.use_wandb:
                wandb.log({
                    'losses/simple_loss': epoch_simple_loss / len(self.train_loader),
                    'losses/vlb_loss': epoch_vlb_loss / len(self.train_loader),
                    'epoch': epoch + 1,
                    'global_step': self.global_step + 1
                })

                if self.debug:
                    wandb.log({
                        'grad_norm': epoch_grad_norm / len(self.train_loader),
                        'epoch': epoch + 1,
                        'global_step': self.global_step + 1
                    })

            if epoch % self.checkpoint_n_epochs == 0:
                self.save_checkpoint(epoch, self.global_step)

        # save a final checkpoint
        self.save_checkpoint(epoch, self.global_step)

    def validate(self):
        logger.info("Running validation...")
        self.model.eval()
        with torch.no_grad():
            # get latents from the LDM using the DDIM sampler. Then use the VAE decoder
            # to get the policies and evaluate their quality
            if self.use_language:
                # get realistic measures to condition on
                gt_params_batch, (measures, text_labels) = next(iter(self.test_loader))
            else:
                gt_params_batch, measures = next(iter(self.test_loader))
                text_labels = None
            measures = measures.to(torch.float32).to(self.device)

            if self.use_language:
                cond = self.model.text_to_cond(text_labels)
                samples = self.sampler.sample(self.model,
                                              shape=[self.test_batch_size, self.latent_channels, self.latent_size,
                                                     self.latent_size],
                                              cond=cond)
            else:
                samples = self.sampler.sample(self.model,
                                              shape=[self.test_batch_size, self.latent_channels, self.latent_size,
                                                     self.latent_size],
                                              cond=measures)

            samples *= (1 / self.scale_factor)
            (rec_policies, rec_obsnorms) = self.autoencoder.decode(samples)

            info = evaluate_agent_quality(self.spec['env'],
                                          self.env,
                                          copy.deepcopy(gt_params_batch),
                                          rec_policies,
                                          rec_obsnorms,
                                          self.test_batch_size,
                                          device=self.device,
                                          normalize_obs=True,
                                          center_data=self.center_data,
                                          weight_normalizer=self.weight_normalizer)
            reward_ratio, js_div = info['Behavior']['reward_ratio'], info['Behavior']['js_div']
            logger.info(f'Reward ratio: {reward_ratio}, JS Divergence: {js_div}')
            return info

    def reevaluate_archive(self, epoch: int, info: Dict[str, Any]):
        '''
        Reevaluate the archive, either using the VAE or the diffusion model itself. Comparing the two allows us to
        see how well the diffusion model's training is progressing. Eventually it should be able to reconstruct the
        archive as well as the VAE
        '''
        logger.info('Evaluating model on entire archive...')

        reconstruction_model = self.model
        if self.use_language:
            # Not enough variety in the language conditions to reconstruct meaningful coverage
            reconstruction_model = None

        # perform reconstruction via subsampling original policies in the archive and seeing if
        # the model can reconstruct them
        subsample_results, image_results = evaluate_ldm_subsample(env_name=self.env_name,
                                                                  archive_df=self.train_archive[0],
                                                                  ldm=reconstruction_model,
                                                                  autoencoder=self.autoencoder,
                                                                  N=-1,
                                                                  image_path=self.save_image_path,
                                                                  suffix=str(epoch),
                                                                  ignore_first=True,
                                                                  sampler=self.sampler,
                                                                  scale_factor=self.scale_factor,
                                                                  normalize_obs=True,
                                                                  clip_obs_rew=self.clip_obs_rew,
                                                                  uniform_sampling=False,
                                                                  cut_out=self.cut_out,
                                                                  average=self.average_elites,
                                                                  latent_shape=(
                                                                  self.z_channels, self.latent_size, self.latent_size),
                                                                  center_data=self.center_data,
                                                                  weight_normalizer=self.weight_normalizer)

        # perform uniform sampling of the original policies in the archive
        uniform_subsample_results, uniform_image_results = evaluate_ldm_subsample(env_name=self.env_name,
                                                                                  archive_df=self.train_archive[0],
                                                                                  ldm=reconstruction_model,
                                                                                  autoencoder=self.autoencoder,
                                                                                  N=-1,
                                                                                  image_path=self.save_image_path,
                                                                                  suffix="uniform_" + str(epoch),
                                                                                  ignore_first=True,
                                                                                  sampler=self.sampler,
                                                                                  scale_factor=self.scale_factor,
                                                                                  normalize_obs=True,
                                                                                  clip_obs_rew=self.clip_obs_rew,
                                                                                  uniform_sampling=True,
                                                                                  cut_out=self.cut_out,
                                                                                  average=self.average_elites,
                                                                                  latent_shape=(
                                                                                      self.z_channels, self.latent_size,
                                                                                      self.latent_size),
                                                                                  center_data=self.center_data,
                                                                                  weight_normalizer=
                                                                                  self.weight_normalizer)

        for key, val in subsample_results['Reconstructed'].items():
            info['Archive/' + key] = val
        for key, val in uniform_subsample_results['Reconstructed'].items():
            info['Archive/Uniform_' + key] = val

        return info, image_results, uniform_image_results