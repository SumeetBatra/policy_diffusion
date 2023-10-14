import torch
import numpy as np
import scipy.stats as stats
import logging

from RL.actor_critic import Actor
from common.tensor_dict import TensorDict
from common.brax_utils import rollout_many_agents
from dataset.shaped_elites_dataset import WeightNormalizer, cat_tensordicts
from typing import Optional

logger = logging.getLogger("metrics")
logger.setLevel(logging.DEBUG)


def kl_divergence(mu1, cov1, mu2, cov2):
    """
    Calculates the KL divergence between two Gaussian distributions.

    Parameters:
    mu1 (numpy array): Mean of the first Gaussian distribution
    cov1 (numpy array): Covariance matrix of the first Gaussian distribution
    mu2 (numpy array): Mean of the second Gaussian distribution
    cov2 (numpy array): Covariance matrix of the second Gaussian distribution

    Returns:
    KL divergence (float)
    """

    # calculate KL divergence using formula
    kl_div = 0.5 * (np.trace(np.linalg.inv(cov2).dot(cov1)) +
                    np.dot((mu2 - mu1).T, np.dot(np.linalg.inv(cov2), (mu2 - mu1))) -
                    len(mu1) + np.log(np.linalg.det(cov2) / (np.linalg.det(cov1) + 1e-9)))

    return kl_div


def js_divergence(mu1, cov1, mu2, cov2):
    '''Jensen-Shannon symmetric divergence metric. It is assumed that all variables
    mu1/2 and cov1/2 parametrize two normal distributions, otherwise this calculation is incorrect'''

    # sum of gaussians is gaussian
    mu_m, cov_m = 0.5 * (mu1 + mu2), 0.5 * (cov1 + cov2)

    res = 0.5 * (kl_divergence(mu1, cov1, mu_m, cov_m) + kl_divergence(mu2, cov2, mu_m, cov_m))
    return res


def calculate_statistics(gt_rews, gt_measures, rec_rewards, rec_measures):
    '''
    Calculate various statistics based on batches of rewards and measures evaluated from the ground truth
    and reconstructed policies
    '''
    gt_mean, gt_cov = gt_measures.mean(0), np.cov(gt_measures.T)
    rec_mean, rec_cov = rec_measures.mean(0), np.cov(rec_measures.T)
    js_div = js_divergence(gt_mean, gt_cov, rec_mean, rec_cov)

    ttest_res = stats.ttest_ind(gt_measures, rec_measures, equal_var=False)

    return {
        'js_div': js_div,
        't_test': ttest_res,
        'measure_mse': np.square(gt_measures.mean(0) - rec_measures.mean(0)),

        'Rewards/original': gt_rews.mean().item(),
        'Measures/original_mean': gt_measures.mean(axis=0),
        'Measures/original_std': gt_measures.std(axis=0),

        'Rewards/reconstructed': rec_rewards.mean().item(),
        'Measures/reconstructed_mean': rec_measures.mean(axis=0),
        'Measures/reconstructed_std': rec_measures.std(axis=0),
    }


def evaluate_agent_quality(env_cfg: dict,
                           vec_env,
                           gt_params_batch: TensorDict,
                           rec_policies: list[Actor],
                           rec_obs_norms: TensorDict,
                           test_batch_size: int,
                           device: str,
                           normalize_obs: bool = False,
                           center_data: bool = False,
                           weight_normalizer: Optional[WeightNormalizer] = None):

    obs_dim = vec_env.single_observation_space.shape[0]
    action_shape = vec_env.single_action_space.shape

    recon_params_batch = [TensorDict(p.state_dict()) for p in rec_policies]
    recon_params_batch = cat_tensordicts(recon_params_batch)
    recon_params_batch.update(rec_obs_norms)

    if center_data:
        assert weight_normalizer is not None and isinstance(weight_normalizer, WeightNormalizer)
        gt_params_batch = weight_normalizer.denormalize(gt_params_batch)
        recon_params_batch = weight_normalizer.denormalize(recon_params_batch)

    if normalize_obs:
        recon_params_batch['obs_normalizer.obs_rms.var'] = torch.exp(recon_params_batch['obs_normalizer.obs_rms.logstd'] * 2)
        recon_params_batch['obs_normalizer.obs_rms.count'] = gt_params_batch['obs_normalizer.obs_rms.count']
        if 'obs_normalizer.obs_rms.logstd' in gt_params_batch:
            del gt_params_batch['obs_normalizer.obs_rms.logstd']
        if 'obs_normalizer.obs_rms.std' in gt_params_batch:
            del gt_params_batch['obs_normalizer.obs_rms.std']
        if 'obs_normalizer.obs_rms.mean' in gt_params_batch:
            del recon_params_batch['obs_normalizer.obs_rms.logstd']


    recon_params_batch['actor_logstd'] = gt_params_batch['actor_logstd']

    gt_agents = [Actor(obs_dim, action_shape, normalize_obs=normalize_obs).to(device) for _ in range(len(gt_params_batch))]
    rec_agents = [Actor(obs_dim, action_shape, normalize_obs=normalize_obs).to(device) for i in range(len(recon_params_batch))]
    for i in range(len(gt_params_batch)):
        gt_agents[i].load_state_dict(gt_params_batch[i])
        rec_agents[i].load_state_dict(recon_params_batch[i])

    # batch-evaluate the ground-truth agents
    gt_rewards, gt_measures = rollout_many_agents(gt_agents, env_cfg, vec_env, device, normalize_obs=normalize_obs, verbose=False)

    # batch-evaluate the reconstructed agents
    rec_rewards, rec_measures = rollout_many_agents(rec_agents, env_cfg, vec_env, device, normalize_obs=normalize_obs, verbose=False)

    # calculate statistics based on results
    info = calculate_statistics(gt_rewards, gt_measures, rec_rewards, rec_measures)
    avg_measure_mse = info['measure_mse']
    avg_t_test = info['t_test'].pvalue
    avg_orig_reward = info['Rewards/original']
    avg_reconstructed_reward = info['Rewards/reconstructed']
    avg_js_div = info['js_div']
    avg_std_orig_measure = info['Measures/original_std']
    avg_std_rec_measure = info['Measures/reconstructed_std']

    reward_ratio = avg_reconstructed_reward / avg_orig_reward

    logger.debug(f'Measure MSE: {avg_measure_mse}')
    logger.debug(f'Reward ratio: {reward_ratio}')
    logger.debug(f'js_div: {avg_js_div}')

    final_info = {
                    'Behavior/measure_mse_0': avg_measure_mse[0],
                    'Behavior/measure_mse_1': avg_measure_mse[1],
                    'Behavior/measure_mse_2': avg_measure_mse[2] if env_cfg.num_dims == 3 else 0,
                    'Behavior/orig_reward': avg_orig_reward,
                    'Behavior/rec_reward': avg_reconstructed_reward,
                    'Behavior/reward_ratio': reward_ratio,
                    'Behavior/p-value_0': avg_t_test[0],
                    'Behavior/p-value_1': avg_t_test[1],
                    'Behavior/p-value_2': avg_t_test[2] if env_cfg.num_dims == 3 else 0,
                    'Behavior/js_div': avg_js_div,
                    'Behavior/std_orig_measure_0': avg_std_orig_measure[0],
                    'Behavior/std_orig_measure_1': avg_std_orig_measure[1],
                    'Behavior/std_orig_measure_2': avg_std_orig_measure[2] if env_cfg.num_dims == 3 else 0,
                    'Behavior/std_rec_measure_0': avg_std_rec_measure[0],
                    'Behavior/std_rec_measure_1': avg_std_rec_measure[1],
                    'Behavior/std_rec_measure_2': avg_std_rec_measure[2] if env_cfg.num_dims == 3 else 0,
                }
    return final_info
