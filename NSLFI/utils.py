import math
from typing import Dict
from typing import Tuple

import anesthetic
import numpy as np
import swyft
import torch
from anesthetic import NestedSamples
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
    index = np.searchsorted(cumulative_weights / cumulative_weights[-1], threshold)
    return index


def compute_KL_divergence(nreSettings: NRE_Settings, previous_network: swyft.SwyftModule,
                          current_samples: anesthetic.Samples, previous_samples: anesthetic.Samples,
                          obs: swyft.Sample) -> Tuple[float, float]:
    """Compute the KL divergence between the previous and current NRE."""

    samples = {nreSettings.targetKey: torch.as_tensor(current_samples.iloc[:, :nreSettings.num_features].to_numpy())}
    with torch.no_grad():
        predictions = previous_network(obs, samples)
    current_samples["logL_previous"] = predictions.logratios.numpy().squeeze()

    if isinstance(current_samples, anesthetic.MCMCSamples):
        # MCMC samples for true samples do not have logw functionality
        posterior = current_samples.iloc[:, :nreSettings.num_features].squeeze()
        true_posterior = nreSettings.model.posterior(obs[nreSettings.obsKey].numpy().squeeze()).logpdf(posterior)
        true_prior = nreSettings.model.prior().logpdf(posterior)
        current_samples.logL = true_posterior
        current_samples["logR"] = current_samples["logL_previous"]
        DKL = (current_samples.logL - current_samples["logR"] - true_prior + previous_samples.logZ()).mean()
        DKL_err = (current_samples.logL - current_samples[
            "logR"] - true_prior + previous_samples.logZ()).std() / np.sqrt(
            len(current_samples.logL))
    else:
        logXPrev = np.interp(x=current_samples["logL_previous"], xp=previous_samples.logL, fp=previous_samples.logX())
        logXnorm_current = logsumexp(-(current_samples.logdX() - current_samples.logX()))
        logXnorm_previous = logsumexp(-(previous_samples.logdX() - previous_samples.logX()))
        current_samples["log_pq"] = (current_samples["logL"] - current_samples.logZ() - current_samples[
            "logL_previous"] + previous_samples.logZ() - current_samples.logX() + logXPrev + logXnorm_current -
                                     logXnorm_previous)
        logw = current_samples.logw(nreSettings.n_DKL_estimates)
        logw -= logsumexp(logw, axis=0)
        DKL_estimates = (np.exp(logw).T * current_samples["log_pq"]).sum(axis=1)
        DKL = DKL_estimates.mean()
        DKL_err = DKL_estimates.std()
    return DKL, DKL_err


def reload_data_for_plotting(nreSettings: NRE_Settings, network: swyft.SwyftModule) -> Tuple[
    Dict[int, str], Dict[int, swyft.SwyftModule]]:
    network_storage = {}
    root_storage = {}
    root = nreSettings.root
    for rd in range(nreSettings.NRE_num_retrain_rounds + 1):
        current_root = f"{root}_round_{rd}"
        new_network = network.get_new_network()
        new_network.load_state_dict(torch.load(f"{current_root}/{nreSettings.neural_network_file}"))
        new_network.double()  # change to float64 precision of network
        network_storage[rd] = new_network
        root_storage[rd] = current_root
    return root_storage, network_storage


def reformat_obs_to_nre_format(obs: swyft.Sample, nreSettings: NRE_Settings) -> Dict[str, torch.Tensor]:
    return {nreSettings.obsKey: torch.tensor(obs[nreSettings.obsKey]).unsqueeze(0)}


def get_swyft_dataset_fractions(fractions, N):
    fractions = np.array(fractions)
    fractions /= sum(fractions)
    mu = N * fractions
    n = np.floor(mu)
    n[0] += N - sum(n)
    return [int(v) for v in n]
