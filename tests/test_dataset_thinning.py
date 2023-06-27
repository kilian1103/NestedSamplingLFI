import math

import torch

from NSLFI.utils import random_subset


def test_dataset_thinning():
    nsize = 100
    ndim = 2
    thinning_factor = 0.314
    dataset = torch.distributions.uniform.Uniform(low=0, high=1).sample(sample_shape=(nsize, ndim))
    subset = random_subset(dataset, thinning_factor)
    assert subset.shape[0] == math.floor(thinning_factor * nsize)


def test_dataset_thinning_with_invalid_factor():
    nsize = 100
    ndim = 2
    thinning_factor = -0.314
    dataset = torch.distributions.uniform.Uniform(low=0, high=1).sample(sample_shape=(nsize, ndim))
    try:
        subset = random_subset(dataset, thinning_factor)
    except ValueError:
        assert True
    else:
        assert False


def test_dataset_thinning_with_wrong_type():
    nsize = 100
    ndim = 2
    thinning_factor = "0.314"
    dataset = torch.distributions.uniform.Uniform(low=0, high=1).sample(sample_shape=(nsize, ndim))
    try:
        subset = random_subset(dataset, thinning_factor)
    except TypeError:
        assert True
    else:
        assert False
