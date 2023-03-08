import os

import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal

from NSLFI.NestedSampler import nested_sampling


def test_nested_sampler_metropolis():
    """
    Test Nested sampling run on Gaussian likelihood using a Metropolis sampler.
    The analytical result for this problem with an uniform (0,1) prior is log Z = 0 for any dimension n.
    """
    np.random.seed(234)

    nlive = 100
    ndim = 5

    def logLikelihood(x) -> np.ndarray:
        # Multivariate Gaussian centred at X = 0.5, y= 0.5

        means = 0.5 * np.ones(shape=ndim)
        cov = 0.01 * np.eye(N=ndim)
        return multivariate_normal.logpdf(x=x, mean=means, cov=cov)

    priors = {f"theta_{i}": stats.uniform(loc=0, scale=1) for i in range(ndim)}
    livepoints = priors["theta_0"].rvs(size=(nlive, ndim))

    logZ = nested_sampling(logLikelihood=logLikelihood, prior=priors,
                           nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype="Metropolis",
                           keep_chain=False, rounds=0)
    print(logZ)
    np.testing.assert_almost_equal(actual=logZ["log Z mean"], desired=0, decimal=0.2)
    os.remove("logL.npy")
    os.remove("logL_birth.npy")
    os.remove("newPoints.npy")
    os.remove("posterior_samples.npy")
    os.remove("weights.npy")


def test_nested_sampler_rejection():
    """
    Test Nested sampling run on Gaussian likelihood using a Rejection sampler.
    The analytical result for this problem with an uniform (0,1) prior is log Z = 0 for any dimension n.
    """
    np.random.seed(234)

    ndim = 2
    nlive = 100

    def logLikelihood(x) -> np.ndarray:
        # Multivariate Gaussian centred at X = 0.5, y= 0.5
        means = 0.5 * np.ones(shape=ndim)
        cov = 0.01 * np.eye(N=ndim)
        return multivariate_normal.logpdf(x=x, mean=means, cov=cov)

    priors = {f"theta_{i}": stats.uniform(loc=0, scale=1) for i in range(ndim)}
    livepoints = priors["theta_0"].rvs(size=(nlive, ndim))

    logZ = nested_sampling(logLikelihood=logLikelihood, prior=priors,
                           nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype="Rejection")
    print(logZ)
    np.testing.assert_almost_equal(actual=logZ["log Z mean"], desired=0, decimal=0.2)
    os.remove("logL.npy")
    os.remove("logL_birth.npy")
    os.remove("newPoints.npy")
    os.remove("posterior_samples.npy")
    os.remove("weights.npy")


def test_nested_sampler_slice():
    """
    Test Nested sampling run on Gaussian likelihood using a slice sampler.
    The analytical result for this problem with an uniform (0,1) prior is log Z = 0 for any dimension n.
    """
    np.random.seed(234)

    ndim = 5
    nlive = 100

    def logLikelihood(x) -> np.ndarray:
        # Multivariate Gaussian centred at X = 0.5, y= 0.5
        means = 0.5 * np.ones(shape=ndim)
        cov = 0.01 * np.eye(N=ndim)
        return multivariate_normal.logpdf(x=x, mean=means, cov=cov)

    priors = {f"theta_{i}": stats.uniform(loc=0, scale=1) for i in range(ndim)}
    livepoints = priors["theta_0"].rvs(size=(nlive, ndim))

    logZ = nested_sampling(logLikelihood=logLikelihood, prior=priors,
                           nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype="Slice", keep_chain=False,
                           rounds=0)
    print(logZ)
    np.testing.assert_almost_equal(actual=logZ["log Z mean"], desired=0, decimal=0.2)
    os.remove("logL.npy")
    os.remove("logL_birth.npy")
    os.remove("newPoints.npy")
    os.remove("posterior_samples.npy")
    os.remove("weights.npy")
