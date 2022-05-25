import numpy as np
import scipy.special
from scipy.stats import multivariate_normal

from Sampler import Sampler


def logLikelihood(x, ndim) -> np.ndarray:
    # Multivariate Gaussian centred at X = 0.5, y= 0.5
    means = 0.5 * np.ones(shape=ndim)
    cov = 0.01 * np.eye(N=ndim)
    return multivariate_normal.logpdf(x=x, mean=means, cov=cov)


def prior(ndim, nsamples) -> np.ndarray:
    return np.random.uniform(low=0, high=1, size=(nsamples, ndim))


def nested_sampling(logLikelihood, prior, ndim, nlive, nsim, stop_criterion, samplertype):
    # initialisation
    logZ_previous = -1e300 * np.ones(nsim)  # Z = 0
    logX_previous = np.zeros(nsim)  # X = 1
    iteration = 0
    logIncrease = 10  # evidence increase factor

    # sample from prior
    print(f"Sampling {nlive} livepoints from the prior!")
    livepoints = prior(ndim, nlive)
    logLikelihoods = logLikelihood(livepoints, ndim)
    livepoints = livepoints.tolist()
    logLikelihoods = logLikelihoods.tolist()
    deadpoints = []
    deadpoints_logL = []
    weights = []

    while logIncrease > np.log(stop_criterion):
        iteration += 1
        # identifying lowest likelihood point
        minlogLike = min(logLikelihoods)
        index = logLikelihoods.index(minlogLike)

        # save deadpoint and its loglike
        deadpoint = livepoints[index]
        deadpoints.append(deadpoint)
        deadpoints_logL.append(minlogLike)

        # sample t's
        ti_s = np.random.power(a=nlive, size=nsim)
        log_ti_s = np.log(ti_s)

        # Calculate X contraction and weight
        logX_current = logX_previous + log_ti_s
        subtraction_coeff = np.array([1, -1]).reshape(2, 1)
        logWeights = np.array([logX_previous, logX_current])
        logWeight_current = scipy.special.logsumexp(a=logWeights, b=subtraction_coeff, axis=0)
        logX_previous = logX_current
        weights.append(np.mean(logWeight_current))

        # Calculate evidence increase
        logZ_current = logWeight_current + minlogLike
        logZ_array = np.array([logZ_previous, logZ_current])
        logZ_total = scipy.special.logsumexp(logZ_array, axis=0)
        logZ_previous = logZ_total

        # find new sample satisfying likelihood constraint
        sampler = Sampler(prior=prior, logLikelihood=logLikelihood, ndim=ndim).getSampler(samplertype)
        proposal_sample = sampler.sample(samples=livepoints, minlogLike=minlogLike)

        # replace lowest likelihood sample with proposal sample
        livepoints[index] = proposal_sample.tolist()
        logLikelihoods[index] = float(logLikelihood(proposal_sample, ndim))

        maxlogLike = max(logLikelihoods)
        logIncrease_array = logWeight_current + maxlogLike - logZ_total
        logIncrease = max(logIncrease_array)
        if iteration % 500 == 0:
            print("current iteration: ", iteration)

    # final <L>*dX sum calculation
    finallogLikesum = scipy.special.logsumexp(a=logLikelihoods)
    logZ_current = -np.log(nlive) + finallogLikesum + logX_current
    logZ_array = np.array([logZ_previous, logZ_current])
    logZ_total = scipy.special.logsumexp(logZ_array, axis=0)

    # convert surviving livepoints to deadpoints
    while len(logLikelihoods) > 0:
        minlogLike = min(logLikelihoods)
        index = logLikelihoods.index(minlogLike)

        deadpoint = livepoints.pop(index)
        logLikelihoods.pop(index)

        deadpoints.append(deadpoint)
        deadpoints_logL.append(minlogLike)
        weights.append(np.mean(logX_current) - np.log(nlive))

    print(f"Algorithm terminated after {iteration} iterations!")
    return {"log Z mean": np.mean(logZ_total),
            "log Z std": np.std(logZ_total)}


logZ = nested_sampling(logLikelihood=logLikelihood, prior=prior, ndim=2, nlive=1000, nsim=100, stop_criterion=1e-3,
                       samplertype="Metropolis")
print(logZ)
