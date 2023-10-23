import logging
import torch
import copy

import wandb
from torch import nn as nn

from trainers.trainer_base import TrainerBase
from common.metrics import evaluate_agent_quality
from common.analysis import evaluate_vae_subsample
from common.tensor_dict import TensorDict
from common.utils import grad_norm
from typing import Dict, Any, Optional, Mapping
from losses.loss_functions import mse_loss_from_weights_dict


logger = logging.getLogger("vae")
logger.setLevel(logging.DEBUG)


class VAETrainer(TrainerBase):
    conditional: bool

    kl_coef: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.info(
            f'Total number of parameters in the encoder: {sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)}')
        logger.info(
            f'Total number of parameters in policy the decoder: {sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)}')
        logger.info(
            f'Total number of parameters in obs_norm decoder {sum(p.numel() for p in self.model.obsnorm_decoder.parameters() if p.requires_grad)}')
        logger.info(f'Total number of paramers:{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def train(self):
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
                    wandb.log({'Archive/recon_image': wandb.Image(image_results['Reconstructed'],
                                                                  caption=f"Epoch {epoch + 1}")})
                    wandb.log({'Archive/Uniform_recon_image': wandb.Image(uniform_image_results['Reconstructed'],
                                                                          caption=f"Epoch {epoch + 1}")})

        # now the main training loop begins
        epoch_mse_loss = 0
        epoch_kl_loss = 0
        epoch_norm_mse_loss = 0
        epoch_grad_norm = 0
        loss_infos = []

        self.model.train()
        for step, (policies, measures) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            measures = measures.to(self.device).to(torch.float32)

            if self.conditional:
                (rec_policies, rec_obsnorms), posterior = self.model(policies, measures)
            else:
                (rec_policies, rec_obsnorms), posterior = self.model(policies)

            rec_obsnorms = TensorDict(rec_obsnorms)
            rec_state_dicts = {}
            for agent in rec_policies:
                for name, param in agent.named_parameters():
                    if name not in rec_state_dicts:
                        rec_state_dicts[name] = []
                    rec_state_dicts[name].append(param)
            for name, param in rec_state_dicts.items():
                rec_state_dicts[name] = torch.stack(param, dim=0)

            # compute loss
            policy_mse_loss, loss_info = mse_loss_from_weights_dict(policies, rec_state_dicts)
            kl_loss = posterior.kl().mean()
            loss = policy_mse_loss + self.kl_coef * kl_loss

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.global_step += 1

            epoch_mse_loss += loss_info['mse_loss']
            epoch_kl_loss += kl_loss.item()
            epoch_norm_mse_loss += loss_info['obsnorm_loss']
            loss_infos.append(loss_info)
            if self.debug:
                epoch_grad_norm += grad_norm(self.model)

        logger.info(f'Epoch: {epoch}, MSE Loss: {epoch_mse_loss / len(self.train_loader)}, ObsNorm MSE Loss:'
                    f' {epoch_norm_mse_loss / len(self.train_loader)}')

        if self.use_wandb:
            avg_loss_infos = {key: sum([loss_info[key] for loss_info in loss_infos]) / len(loss_infos) for key in loss_infos[0].keys()}

            self.writer.add_scalar("Loss/mse_loss", epoch_mse_loss / len(self.train_loader), self.global_step + 1)
            self.writer.add_scalar("Loss/kl_loss", epoch_kl_loss / len(self.train_loader), self.global_step + 1)
            self.writer.add_scalar("Loss/norm_mse_loss", epoch_norm_mse_loss / len(self.train_loader), self.global_step + 1)

            wandb.log({
                'Loss/mse_loss': epoch_mse_loss / len(self.train_loader),
                'Loss/kl_loss': epoch_kl_loss / len(self.train_loader),
                'Loss/norm_mse_loss': epoch_norm_mse_loss / len(self.train_loader),
                'grad_norm': epoch_grad_norm / len(self.train_loader),
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
        self.model.eval()
        with torch.no_grad():
            # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
            # performance and behavior to the ground truth
            gt_params, gt_measure = next(iter(self.test_loader))
            gt_measure = gt_measure.to(self.device).to(torch.float32)

            if self.conditional:
                (rec_policies, rec_obsnorm), _ = self.model(gt_params, gt_measure)
            else:
                (rec_policies, rec_obsnorms), _ = self.model(gt_params)
            rec_obsnorms = TensorDict(rec_obsnorms)

            info = evaluate_agent_quality(self.spec['env'],
                                          self.env,
                                          gt_params,
                                          rec_policies,
                                          rec_obsnorms,
                                          self.test_batch_size,
                                          device=self.device,
                                          normalize_obs=True,
                                          center_data=self.spec['dataset']['center_data'],
                                          weight_normalizer=self.spec['dataset']['weight_normalizer'])

            # now try to sample a policy with just measures
            if self.conditional:
                (rec_policies, rec_obsnorms), _ = self.model(None, gt_measure)

                info2 = evaluate_agent_quality(self.spec['env'],
                                               self.env,
                                               gt_params,
                                               rec_policies,
                                               rec_obsnorms,
                                               self.test_batch_size,
                                               device=self.device,
                                               normalize_obs=True,
                                               center_data=self.spec['dataset']['center_data'],
                                               weight_normalizer=self.spec['dataset']['weight_normalizer'])

                for key, val in info2.items():
                    info['Conditional_' + key] = val

        return info

    def reevaluate_archive(self, epoch: int, info: Dict[str, Any]):
        '''
        Reevaluate the archive, either using the VAE or the diffusion model itself. Comparing the two allows us to
        see how well the diffusion model's training is progressing. Eventually it should be able to reconstruct the
        archive as well as the VAE
        '''

        subsample_results, image_results = evaluate_vae_subsample(env_name=self.env_name,
                                                                  archive_df=self.train_archive[0],
                                                                  model=self.model,
                                                                  N=-1,
                                                                  image_path=self.save_image_path,
                                                                  suffix=str(epoch),
                                                                  ignore_first=True,
                                                                  normalize_obs=True,
                                                                  average=self.average_elites,
                                                                  clip_obs_rew=self.clip_obs_rew,
                                                                  center_data=self.spec['dataset']['center_data'],
                                                                  weight_normalizer=self.spec['dataset'][
                                                                      'weight_normalizer']
                                                                  )

        for key, val in subsample_results['Reconstructed'].items():
            info['Archive/' + key] = val

        return info, image_results, subsample_results
