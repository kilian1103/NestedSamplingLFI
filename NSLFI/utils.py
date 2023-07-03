import math

import numpy as np
from anesthetic import NestedSamples
from torch import Tensor


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
