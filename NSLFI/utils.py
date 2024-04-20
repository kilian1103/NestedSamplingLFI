import math
from typing import Dict
from typing import Tuple

import anesthetic
import numpy as np
import swyft
import torch
from anesthetic import NestedSamples
from pypolychord import PolyChordSettings
from scipy.special import logsumexp
from torch import Tensor

from NSLFI.NRE_Settings import NRE_Settings


def random_subset(dataset: Tensor, thinning_factor: float) -> Tensor:
    """
    Randomly select a subset of the dataset
    :param dataset: torch.Tensor of shape (nsize, ndim)
    :param thinning_factor: float between 0 and 1
    :return:
    """
    if not isinstance(thinning_factor, int) and not isinstance(thinning_factor, float):
        raise TypeError("thinning_factor must be a float or int type")
    if thinning_factor > 1 or thinning_factor <= 0:
        raise ValueError("thinning_factor must be between 0 and 1")
    N = dataset.shape[0]  # Size of the dataset
    num_samples = math.floor(thinning_factor * N)  # Number of samples in the random subset
    if num_samples == 0:
        raise ValueError("thinning_factor is too small")
    indices = np.random.choice(N, num_samples, replace=False)  # Randomly select indices
    return dataset[indices]  # Return the subset of the dataset


def select_weighted_contour(data: NestedSamples, threshold: float) -> int:
    """find the index of the sample that corresponds to the threshold of the cumulative weights."""
    cumulative_weights = data.get_weights().cumsum()
    cumulative_weights_norm = cumulative_weights / cumulative_weights[-1]
    index = np.searchsorted(cumulative_weights_norm, threshold)
    return index


def compute_KL_divergence(nreSettings: NRE_Settings, previous_network: swyft.SwyftModule,
                          current_samples: anesthetic.Samples, previous_samples: anesthetic.Samples,
                          obs: swyft.Sample) -> Tuple[float, float]:
    """Compute the KL divergence between the previous and current NRE."""

    samples = {nreSettings.targetKey: torch.as_tensor(current_samples.iloc[:, :nreSettings.num_features].to_numpy())}
    with torch.no_grad():
        predictions = previous_network(obs, samples)
    current_samples["logL_previous"] = predictions.logratios.numpy().squeeze()

    logw = current_samples.logw(nreSettings.n_DKL_estimates)
    logpqs = (current_samples["logL"].values[:, None] - current_samples.logZ(logw).values - current_samples[
                                                                                                "logL_previous"].values[
                                                                                            :,
                                                                                            None] +
              previous_samples.logZ(
                  nreSettings.n_DKL_estimates).values)
    logw -= logsumexp(logw, axis=0)
    DKL_estimates = (np.exp(logw).T * logpqs.T).sum(axis=1)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()

    return DKL, DKL_err


def compute_KL_compression(samples: anesthetic.NestedSamples, nreSettings: NRE_Settings):
    """Compute the Prior to Posterior DKL compression"""
    logw = samples.logw(nreSettings.n_DKL_estimates)
    logpqs = samples["logL"].values[:, None] - samples.logZ(logw).values
    logw -= logsumexp(logw, axis=0)
    DKL_estimates = (np.exp(logw).T * logpqs.T).sum(axis=1)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()
    return DKL, DKL_err


def reload_data_for_plotting(nreSettings: NRE_Settings, network: swyft.SwyftModule, polyset: PolyChordSettings,
                             until_round: int, only_last_round=False) -> \
        Tuple[
            Dict[int, str], Dict[int, swyft.SwyftModule], Dict[int, anesthetic.NestedSamples]]:
    network_storage = {}
    root_storage = {}
    samples_storage = {}
    root = nreSettings.root

    for rd in range(until_round + 1):
        # load root
        if only_last_round and rd < until_round:
            continue

        current_root = f"{root}_round_{rd}"
        root_storage[rd] = current_root

        # load network
        new_network = network.get_new_network()
        new_network.load_state_dict(torch.load(f"{current_root}/{nreSettings.neural_network_file}"))
        new_network.double()  # change to float64 precision of network
        network_storage[rd] = new_network

        # load samples
        params = [fr"${nreSettings.targetKey}_{i}$" for i in range(nreSettings.num_features)]
        if nreSettings.use_livepoint_increasing:
            samples = anesthetic.read_chains(
                root=f"{root_storage[rd]}/{nreSettings.increased_livepoints_fileroot}/{polyset.file_root}")
        else:
            samples = anesthetic.read_chains(root=f"{root_storage[rd]}/{polyset.file_root}")
        labels = samples.get_labels()
        labels[:nreSettings.num_features] = params
        samples.set_labels(labels, inplace=True)
        samples_storage[rd] = samples

    return root_storage, network_storage, samples_storage


def reformat_obs_to_nre_format(obs: swyft.Sample, nreSettings: NRE_Settings) -> Dict[str, torch.Tensor]:
    return {nreSettings.obsKey: torch.tensor(obs[nreSettings.obsKey]).unsqueeze(0)}
