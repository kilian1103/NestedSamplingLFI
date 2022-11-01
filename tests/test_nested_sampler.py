import numpy as np
from scipy.stats import multivariate_normal

from NSLFI.NestedSampler import nested_sampling


def test_nested_sampler():
    np.random.seed(234)

    def logLikelihood(x, ndim) -> np.ndarray:
        # Multivariate Gaussian centred at X = 0.5, y= 0.5
        means = 0.5 * np.ones(shape=ndim)
        cov = 0.01 * np.eye(N=ndim)
        return multivariate_normal.logpdf(x=x, mean=means, cov=cov)

    def prior(ndim, nsamples, limits) -> np.ndarray:
        return np.random.uniform(low=limits["lower"], high=limits["upper"], size=(nsamples, ndim))

    ndim = 2
    nlive = 100
    priorLimits = {"lower": np.zeros(ndim),
                   "upper": np.ones(ndim)}

    livepoints = prior(ndim=ndim, nsamples=nlive, limits=priorLimits)
    logZ = nested_sampling(logLikelihood=logLikelihood, prior=prior, priorLimits=priorLimits, ndim=ndim,
                           nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype="Metropolis")
    print(logZ)
    np.testing.assert_almost_equal(actual=logZ["log Z mean"], desired=0, decimal=0.2)
