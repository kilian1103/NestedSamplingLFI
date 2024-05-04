import os
from typing import Dict
from typing import Tuple

import anesthetic
import numpy as np
import pandas as pd
import swyft
import torch
from anesthetic import NestedSamples
from pypolychord import PolyChordSettings
from scipy.special import logsumexp

from NSLFI.NRE_Settings import NRE_Settings


def select_weighted_contour(data: NestedSamples, threshold: float) -> int:
    """find the index of the posterior sample that corresponds iso-contour threshold.
    :param data: An anesthetic NestedSamples object
    :param threshold: A float between 0 and 1
    :return: An integer index
    """
    cumulative_weights = data.get_weights().cumsum()
    cumulative_weights_norm = cumulative_weights / cumulative_weights[-1]
    index = np.searchsorted(cumulative_weights_norm, threshold)
    return index


def compute_KL_divergence(nreSettings: NRE_Settings, previous_network: swyft.SwyftModule,
                          current_samples: anesthetic.Samples, previous_samples: anesthetic.Samples,
                          obs: swyft.Sample) -> Tuple[float, float]:
    """
    Compute the KL divergence between the previous NRE and the current NRE KL(P_{i}||P_{i-i}).
    :param nreSettings: A NRE_Settings object
    :param previous_network: A swyft network object
    :param current_samples: An anesthetic samples object of the current samples
    :param previous_samples: An anesthetic samples object of the previous samples
    :param obs: A swyft sample of the observed data
    :return: A tuple of the KL divergence and the error
    """

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


def compute_KL_divergence_truth(nreSettings: NRE_Settings, previous_network: swyft.SwyftModule,
                                true_posterior: anesthetic.Samples, previous_samples: anesthetic.Samples,
                                obs: swyft.Sample) -> Tuple[float, float]:
    """Compute the KL divergence between the previous NRE and the true posterior KL(P_{true}||P_{i}).
    :param nreSettings: A NRE_Settings object
    :param previous_network: A swyft network object
    :param true_posterior: An anesthetic samples object of the true posterior
    :param previous_samples: An anesthetic samples object of the previous samples
    :param obs: A swyft sample of the observed data
    :return: A tuple of the KL divergence and the error
    """
    swyft_samples = {
        nreSettings.targetKey: torch.as_tensor(true_posterior.iloc[:, :nreSettings.num_features].to_numpy())}
    with torch.no_grad():
        predictions = previous_network(obs, swyft_samples)
    true_posterior["logL_previous"] = predictions.logratios.numpy().squeeze()
    # MCMC samples for true samples do not have logw functionality
    samples = true_posterior.iloc[:, :nreSettings.num_features].squeeze()
    true_posterior_logL = nreSettings.model.posterior(obs[nreSettings.obsKey].numpy().squeeze()).logpdf(samples)
    true_prior = nreSettings.model.prior().logpdf(samples)
    true_posterior.logL = true_posterior_logL
    true_posterior["logR"] = true_posterior["logL_previous"]
    logpqs = (true_posterior["logL"].values[:, None] - true_posterior["logR"].values[:, None] - true_prior[:,
                                                                                                None] +
              previous_samples.logZ(
                  nreSettings.n_DKL_estimates).values)
    DKL_estimates = logpqs.mean(axis=0)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()
    return (DKL, DKL_err)


def compute_KL_compression(samples: anesthetic.NestedSamples, nreSettings: NRE_Settings):
    """
    Compute the KL compression of the samples, Prior to Posterior, KL(P||pi).
    :param samples: An anesthetic NestedSamples object
    :param nreSettings: A NRE_Settings object
    :return: A tuple of the KL compression and the error
    """
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
            Dict[int, str], Dict[int, swyft.SwyftModule], Dict[int, anesthetic.NestedSamples], Dict[
                int, Tuple[float, float]]]:
    """
    Reload the data for plotting.
    :param nreSettings: A NRE_Settings object
    :param network: A swyft network object
    :param polyset: A PolyChordSettings object
    :param until_round: An integer of the number of rounds to reload (inclusive)
    :param only_last_round: A boolean to only reload the last round until_round
    :return: A tuple of dictionaries of root_storage, network_storage, samples_storage, and dkl_storage
    """

    network_storage = {}
    root_storage = {}
    samples_storage = {}
    dkl_storage = {}
    root = nreSettings.root

    try:
        obs = network.obs
    except AttributeError:
        raise AttributeError("network object does not have an attribute 'obs'")

    for rd in range(until_round + 1):
        if only_last_round and rd < until_round - 1:
            continue

        current_root = f"{root}/{nreSettings.child_root}_{rd}"
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

        # compute DKL
        if rd > 0:
            if only_last_round and rd < until_round:
                continue
            previous_network = network_storage[rd - 1]
            KDL = compute_KL_divergence(nreSettings=nreSettings, previous_network=previous_network.eval(),
                                        current_samples=samples_storage[rd], obs=obs,
                                        previous_samples=samples_storage[rd - 1])
            dkl_storage[rd] = KDL
    return root_storage, network_storage, samples_storage, dkl_storage


def random_subset_after_truncation(deadpoints: anesthetic.NestedSamples, logR_cutoff: float,
                                   p: float) -> anesthetic.NestedSamples:
    rest = deadpoints[deadpoints.logL >= logR_cutoff]
    bools = np.random.choice([True, False], size=rest.shape[0], p=[p, 1 - p])
    rest = rest[bools]
    deadpoints = pd.concat([deadpoints, rest], axis=0)
    deadpoints.drop_duplicates(inplace=True)
    return deadpoints


def delete_previous_joint_training_data(until_rd: int, root: str, nreSettings: NRE_Settings):
    """
    Delete previous joint training data.
    :param until_rd: An integer of the round number to delete until
    :param root: A string of the root directory
    """
    for rd in range(until_rd):
        try:
            os.remove(f"{root}/{nreSettings.child_root}_{rd}/{nreSettings.joint_training_data_fileroot}")
        except FileNotFoundError:
            pass
    return
