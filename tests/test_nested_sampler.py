import os

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from NSLFI.NestedSampler import nested_sampling


def test_nested_sampler_metropolis():
    """
    Test Nested sampling run on Gaussian likelihood using a Metropolis sampler.
    The analytical result for this problem with an uniform (0,1) prior is log Z = 0 for any dimension n.
    """
    torch.random.manual_seed(234)
    np.random.seed(234)

    ndim = 5
    nlive = 25 * ndim

    means = 0.5 * torch.ones(ndim)
    cov = 0.01 * torch.eye(ndim)
    mvNormal = MultivariateNormal(loc=means, covariance_matrix=cov)

    def logLikelihood(x) -> torch.tensor:
        return mvNormal.log_prob(x)

    priors = {f"theta_{i}": torch.distributions.uniform.Uniform(low=0, high=1) for i in range(ndim)}
    livepoints = priors["theta_0"].sample(sample_shape=(nlive, ndim)).type(torch.float64)

    logZ = nested_sampling(logLikelihood=logLikelihood, prior=priors,
                           nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype="Metropolis",
                           keep_chain=False)
    print(logZ)
    torch.testing.assert_close(actual=logZ["log Z mean"], expected=0., atol=0.3, rtol=0.2)
    os.remove("logL")
    os.remove("logL_birth")
    os.remove("posterior_samples")
    os.remove("weights")


def test_nested_sampler_rejection():
    """
    Test Nested sampling run on Gaussian likelihood using a Rejection sampler.
    The analytical result for this problem with an uniform (0,1) prior is log Z = 0 for any dimension n.
    """
    torch.random.manual_seed(234)
    np.random.seed(234)

    ndim = 2
    nlive = 25 * ndim

    means = 0.5 * torch.ones(ndim)
    cov = 0.01 * torch.eye(ndim)
    mvNormal = MultivariateNormal(loc=means, covariance_matrix=cov)

    def logLikelihood(x) -> torch.tensor:
        return mvNormal.log_prob(x)

    priors = {f"theta_{i}": torch.distributions.uniform.Uniform(low=0, high=1) for i in range(ndim)}
    livepoints = priors["theta_0"].sample(sample_shape=(nlive, ndim)).type(torch.float64)

    logZ = nested_sampling(logLikelihood=logLikelihood, prior=priors,
                           nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype="Rejection")
    print(logZ)
    torch.testing.assert_close(actual=logZ["log Z mean"], expected=0., atol=0.3, rtol=0.2)
    os.remove("logL")
    os.remove("logL_birth")
    os.remove("posterior_samples")
    os.remove("weights")


def test_nested_sampler_slice():
    """
    Test Nested sampling run on Gaussian likelihood using a slice sampler.
    The analytical result for this problem with an uniform (0,1) prior is log Z = 0 for any dimension n.
    """
    torch.random.manual_seed(234)
    np.random.seed(234)

    ndim = 5
    nlive = 25 * ndim

    means = 0.5 * torch.ones(ndim)
    cov = 0.01 * torch.eye(ndim)
    mvNormal = MultivariateNormal(loc=means, covariance_matrix=cov)

    def logLikelihood(x) -> torch.tensor:
        return mvNormal.log_prob(x)

    priors = {f"theta_{i}": torch.distributions.uniform.Uniform(low=0, high=1) for i in range(ndim)}
    livepoints = priors["theta_0"].sample(sample_shape=(nlive, ndim)).type(torch.float64)

    logZ = nested_sampling(logLikelihood=logLikelihood, prior=priors,
                           nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype="Slice", keep_chain=False)
    print(logZ)
    torch.testing.assert_close(actual=logZ["log Z mean"], expected=0., atol=0.3, rtol=0.2)
    os.remove("logL")
    os.remove("logL_birth")
    os.remove("posterior_samples")
    os.remove("weights")
