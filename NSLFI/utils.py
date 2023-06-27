import math

import numpy as np
import torch


def random_subset(dataset: torch.Tensor, thinning_factor: float):
    """
    Randomly select a subset of the dataset
    :param dataset: torch.Tensor of shape (nsize, ndim)
    :param thinning_factor: float between 0 and 1
    :return:
    """
    if isinstance(thinning_factor, str):
        raise TypeError("thinning_factor must be a float or int type and not a string")
    if thinning_factor > 1 or thinning_factor <= 0:
        raise ValueError("thinning_factor must be between 0 and 1")
    N = dataset.shape[0]  # Size of the dataset
    num_samples = math.floor(thinning_factor * N)  # Number of samples in the random subset
    indices = np.random.choice(N, num_samples, replace=False)  # Randomly select indices
    return dataset[indices]  # Return the subset of the dataset
