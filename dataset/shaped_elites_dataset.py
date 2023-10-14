import pickle
import pandas
import torch
import os
import glob
import numpy as np
import logging

from typing import List, Union, Optional, Dict, Any
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from RL.actor_critic import Actor
# from ribs.archives._elite import EliteBatch
from tqdm import tqdm
from RL.normalize import ObsNormalizer
from common.tensor_dict import TensorDict, cat_tensordicts
from common.brax_utils import shared_params
from common.archive_utils import archive_df_to_archive


logger = logging.getLogger("dataset")
logger.setLevel(logging.DEBUG)


def readonly(arr):
    """Sets an array to be readonly."""
    arr.flags.writeable = False
    return arr


class WeightNormalizer:
    def __init__(self, means_dict: TensorDict, std_dict: TensorDict):
        '''
        A class to normalize the weights of a policy across all policies in the dataset
        :param means_dict: dict of means for each layer
        :param std_dict: dict of stds for each layer
        '''
        self.means_dict = means_dict
        self.std_dict = std_dict

    def normalize(self, data: TensorDict):
        for name, param in data.items():
            data[name] = (param - self.means_dict[name]) / (self.std_dict[name] + 1e-8)

        return data

    def denormalize(self, data: TensorDict):
        for name, param in data.items():
            data[name] = param * (self.std_dict[name] + 1e-8) + self.means_dict[name]

        return data

    def state_dict(self):
        return {
            'means_dict': dict(self.means_dict),
            'std_dict': dict(self.std_dict)
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.means_dict = TensorDict(state_dict['means_dict'])
        self.std_dict = TensorDict(state_dict['std_dict'])


class ShapedEliteDataset(Dataset):
    def __init__(self,
                 archive_dfs: list[DataFrame],
                 obs_dim: int,
                 action_shape: Union[tuple, np.ndarray],
                 device: str,
                 is_eval: bool = False,
                 eval_batch_size: Optional[int] = 8,
                 center_data: bool = False,
                 cut_out: bool = False,
                 weight_normalizer: Optional[WeightNormalizer] = None):
        '''
        Dataset class that takes in the archive of policies (neural networks) and converts them into a
        sliceable dictionary of weights.
        :param archive_dfs: Dataframes containing the policies to distill
        :param obs_dim: The obs (input) dim of the policy
        :param action_shape: The action (output) shape of the policy
        :param device: cuda / cpu
        :param is_eval: For evaluation, subsample the dataset
        :param eval_batch_size: Batch size for evaluation only
        :param center_data: Whether or not to use the weight normalizer to normalize the data
        :param cut_out: Perform the "cut out" experiment where we hold out policies in part of the archive and see if we can generalize
        :param weight_normalizer: WeightNormalizer
        '''
        archive_df = pandas.concat(archive_dfs)

        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.device = device
        self.is_eval = is_eval
        self.cut_out = cut_out

        self.measures_list = archive_df.filter(regex='measure*').to_numpy()
        self.metadata = archive_df.filter(regex='metadata*').to_numpy()
        self.objective_list = archive_df['objective'].to_numpy()

        elites_list = archive_df.filter(regex='solution*').to_numpy()

        if cut_out:
            indices_to_cut = np.argwhere((self.measures_list[:,0] > 0.5) * (self.measures_list[:,1] > 0.5) * (self.measures_list[:,0] < 0.6) * (self.measures_list[:,1] < 0.6))
            elites_list = np.delete(elites_list, indices_to_cut, axis=0)
            self.measures_list = np.delete(self.measures_list, indices_to_cut, axis=0)
            self.metadata = np.delete(self.metadata, indices_to_cut, axis=0)
            self.objective_list = np.delete(self.objective_list, indices_to_cut, axis=0)


        if self.is_eval:
            # indices shall be eval_batch_size number of indices spaced out (by objective) evenly across the elites_list
            indices = np.linspace(0, len(elites_list) - 1, eval_batch_size, dtype=int)
            indices = np.argsort(archive_df['objective'].to_numpy())[indices]
            self.indices = indices
            elites_list = elites_list[indices]
            self.measures_list = self.measures_list[indices]
            self.metadata = self.metadata[indices]
            self.objective_list = self.objective_list[indices]

        self._size = len(elites_list)

        weight_dicts_list = self._params_to_weight_dicts(elites_list)
        self.weights_dict = cat_tensordicts(weight_dicts_list)

        # per-layer mean and std-dev stats for centering / de-centering the data
        if weight_normalizer is None:
            weight_mean_dict = TensorDict({
                key: self.weights_dict[key].mean(0).to(self.device) for key in self.weights_dict.keys()
            })

            weight_std_dict = TensorDict({
                key: self.weights_dict[key].std(0).to(self.device) for key in self.weights_dict.keys()
            })
            weight_normalizer = WeightNormalizer(means_dict=weight_mean_dict, std_dict=weight_std_dict)

        self.normalizer = weight_normalizer

        # zero center the data with unit variance
        if center_data:
            self.weights_dict = self.normalizer.normalize(self.weights_dict)

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        weights_dict, measures = self.weights_dict[item], self.measures_list[item]
        return weights_dict, measures

    def _params_to_weight_dicts(self, elites_list):
        weight_dicts = []
        for i, params in tqdm(enumerate(elites_list)):
            agent = Actor(self.obs_dim, self.action_shape, True, False)
            normalize_obs = self.metadata[i][0]['obs_normalizer']
            if isinstance(normalize_obs, dict):
                obs_normalizer = ObsNormalizer(self.obs_dim).to(self.device)
                obs_normalizer.load_state_dict(normalize_obs)
                agent.obs_normalizer = obs_normalizer
            else:
                agent.obs_normalizer = normalize_obs

            weights_dict = TensorDict(agent.deserialize(params).to(self.device).state_dict())
            weights_dict['obs_normalizer.obs_rms.std'] = torch.sqrt(weights_dict['obs_normalizer.obs_rms.var'] + 1e-8)
            weights_dict['obs_normalizer.obs_rms.logstd'] = torch.log(weights_dict['obs_normalizer.obs_rms.std'])
            weight_dicts.append(weights_dict)
        return weight_dicts


class LangShapedEliteDataset(ShapedEliteDataset):

    def __init__(self, *args, text_labels: List[str], **kwargs):
        '''
        Language-labeled version of the ShapedElites dataset
        '''
        super().__init__(*args, **kwargs)
        self.text_labels = text_labels
        if self.is_eval:
            self.text_labels = [self.text_labels[i] for i in self.indices]

    def __getitem__(self, item):
        weights_dict, measures = super().__getitem__(item)
        return weights_dict, (measures, self.text_labels[item])


def shaped_elites_dataset_factory(env_name,
                                  batch_size=32,
                                  is_eval=False,
                                  center_data: bool = False,
                                  weight_normalizer: Optional[WeightNormalizer] = None,
                                  use_language: bool = False,
                                  results_folder = "results",
                                  N=-1,
                                  cut_out = False,):
    archive_data_path = f'data/{env_name}'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            logger.info(f'Loading archive at {path}')
            archive_df = pickle.load(f)

            if cut_out:
                print('Cutting out the middle of the archive')
                ln_before_cut = len(archive_df)
                # ignore the elites that are in the middle of the archive
                archive_df = archive_df[
                    ~((archive_df['measure_0'] > 0.2) & (archive_df['measure_1'] > 0.2)
                    & (archive_df['measure_0'] < 0.6) & (archive_df['measure_1'] < 0.6))]
                print(f'Cut out {ln_before_cut - len(archive_df)} elites')

            if N != -1:
                archive_df = archive_df.sample(N)


            archive_dfs.append(archive_df)

    if use_language:
        text_label_paths = sorted(glob.glob(archive_data_path + '/text_labels_*.pkl'))
        path = text_label_paths[-1]
        with open(path, 'rb') as f:
            logger.info(f'Loading text labels from {path}')
            text_labels = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obs_dim, action_shape = shared_params[env_name]['obs_dim'], np.array([shared_params[env_name]['action_dim']])

    if is_eval:
        archive_df = pandas.concat(archive_dfs)
        # compute a lower dimensional cvt archive from the original dataset
        soln_dim = archive_df.filter(regex='solution*').to_numpy().shape[1]
        cells = batch_size
        ranges = [(0.0, 1.0)] * shared_params[env_name]['env_cfg']['num_dims']

        # load centroids if they were previously calculated. Maintains consistency across runs
        centroids = None
        centroids_path = f'{results_folder}/{env_name}/centroids.npy'
        if os.path.exists(centroids_path):
            logger.info(f'Existing centroids found at {centroids_path}. Loading centroids...')
            centroids = np.load(centroids_path)
        cvt_archive = archive_df_to_archive(archive_df,
                                            type='cvt',
                                            solution_dim=soln_dim,
                                            cells=cells,
                                            ranges=ranges,
                                            custom_centroids=centroids)
        np.save(centroids_path, cvt_archive.centroids)
        # overload the archive_dfs variable with the new archive_df containing only solutions corresponding to the
        # centroids
        archive_dfs = [cvt_archive.as_pandas(include_solutions=True, include_metadata=True)]

    if use_language:
        s_elite_dataset = LangShapedEliteDataset(archive_dfs, obs_dim=obs_dim,
                                                 action_shape=action_shape,
                                                 device=device,
                                                 is_eval=is_eval,
                                                 eval_batch_size=batch_size if
                                                 is_eval else None,
                                                 center_data=center_data,
                                                 weight_normalizer=weight_normalizer,
                                                 text_labels=text_labels)
    else:
        s_elite_dataset = ShapedEliteDataset(archive_dfs, obs_dim=obs_dim,
                                             action_shape=action_shape,
                                             device=device, is_eval=is_eval,
                                             eval_batch_size=batch_size if
                                             is_eval else None,
                                             center_data=center_data,
                                             weight_normalizer=weight_normalizer,
                                             cut_out=cut_out)

    weight_normalizer = s_elite_dataset.normalizer
    return DataLoader(s_elite_dataset, batch_size=batch_size, shuffle=not is_eval), archive_dfs, weight_normalizer