import math
from typing import Dict
from typing import Tuple

import anesthetic
import numpy as np
import torch
from anesthetic import NestedSamples
from torch import Tensor

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NSNRE_data_generation import DataEnvironment


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


def compute_KL_divergence(nreSettings: NRE_Settings, network_storage: Dict[str, NRE_PolyChord],
                          current_samples: anesthetic.Samples, rd: int) -> Tuple[float, float]:
    """Compute the KL divergence between the previous and current NRE."""
    previous_network = network_storage[f"round_{rd - 1}"]
    with torch.no_grad():
        current_samples["logL_previous"] = \
            previous_network.logLikelihood(current_samples.iloc[:, :nreSettings.num_features].to_numpy())[0]
    DKL = (current_samples["logL"] - current_samples["logL_previous"]).mean()
    DKL_err = (current_samples["logL"] - current_samples["logL_previous"]).std()
    return DKL, DKL_err


def reload_data_for_plotting() -> Tuple[Dict[str, str], Dict[str, NRE_PolyChord], NRE_Settings, DataEnvironment]:
    nreSettings = NRE_Settings()
    dataEnv = DataEnvironment(nreSettings=nreSettings)
    dataEnv.generate_data()
    root = nreSettings.root
    network_storage = {}
    root_storage = {}
    for rd in range(nreSettings.NRE_num_retrain_rounds + 1):
        current_root = f"{root}_round_{rd}"
        current_network = Network(nreSettings=nreSettings)
        current_network.load_state_dict(torch.load(f"{current_root}/{nreSettings.neural_network_file}"))
        current_network.double()  # change to float64 precision of network
        trained_NRE = NRE_PolyChord(network=current_network, obs=dataEnv.obs)
        network_storage[f"round_{rd}"] = trained_NRE
        root_storage[f"round_{rd}"] = current_root
    return root_storage, network_storage, nreSettings, dataEnv
