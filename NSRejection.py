import numpy as np
import scipy.special
from scipy.stats import multivariate_normal


def logLikelihood(x, ndim) -> np.ndarray:
    # Multivariate Gaussian centred at X = 0.5, y= 0.5
    # x shape: (ndim, n_samples)
    means = 0.5 * np.ones(shape=ndim)
    cov = 0.01 * np.eye(N=ndim)
    return multivariate_normal.logpdf(x=x, mean=means, cov=cov)


def prior(ndim, nsamples) -> np.ndarray:
    # random_directions = np.random.normal(silogze=(C,n_samples))
    # norm = np.linalg.norm(random_directions, axis=0)
    # random_directions/=norm
    return np.random.uniform(low=0, high=3, size=(nsamples, ndim))


def rejection_sampler(ndim, prior) -> np.ndarray:
    proposal_sample = prior(ndim, 1)[0]
    return proposal_sample


def metropolis_sampler(ndim, livepoints, nrepeat=5) -> np.ndarray:
    random_index = np.random.randint(0, len(livepoints))
    cov = np.cov(np.array(livepoints).T)
    current_sample = livepoints[random_index]
    for i in range(nrepeat * ndim):
        withinPrior = False
        while withinPrior is False:
            proposal_sample = multivariate_normal.rvs(mean=current_sample, cov=cov)
            # proposal_sample = np.random.normal(loc=livepoints[random_index], scale= 0.0005, size=ndim)
            withinPrior = np.logical_and(proposal_sample > 0, proposal_sample < 4).all()
        current_sample = proposal_sample
    return current_sample


def nested_sampling(logLikelihood, prior, ndim, nlive, stop_criterion, sampler):
    # initialisation
    logZ_previous = -1e300 * np.ones(nlive)  # Z = 0
    logX_previous = np.zeros(nlive)  # X = 1
    iteration = 0
    logIncrease = 10  # evidence increase factor

    # sample from prior
    samples = prior(ndim, nlive)
    logLikelihoods = logLikelihood(samples, ndim)
    samples = samples.tolist()
    logLikelihoods = logLikelihoods.tolist()

    while logIncrease > np.log(stop_criterion):
        iteration += 1
        minlogLike = min(logLikelihoods)
        index = logLikelihoods.index(minlogLike)

        # sample t's
        ti_s = np.amax(np.random.random(size=(nlive, nlive)), axis=0)
        log_ti_s = np.log(ti_s)

        logX_current = logX_previous + log_ti_s

        subtraction_coeff = np.array([1, -1])
        logWeights = np.array([logX_previous, logX_current])
        logWeight_current = scipy.special.logsumexp(a=logWeights.T, b=subtraction_coeff, axis=1)
        logX_previous = logX_current

        logZ_current = logWeight_current + minlogLike
        logZ_array = np.array([logZ_previous, logZ_current])
        logZ_total = scipy.special.logsumexp(logZ_array, axis=0)
        logZ_previous = logZ_total

        sampling = True
        while sampling:
            # proposal_sample = sampler(ndim, samples)
            proposal_sample = sampler(ndim, prior)
            if logLikelihood(proposal_sample, ndim) > minlogLike:
                # accept
                samples[index] = proposal_sample.tolist()
                logLikelihoods[index] = float(logLikelihood(proposal_sample, ndim))
                sampling = False

        maxlogLike = max(logLikelihoods)
        logIncrease_array = logWeight_current + maxlogLike - logZ_total
        logIncrease = min(logIncrease_array)
        if iteration % 1000 == 0:
            print("current iteration: ", iteration)
            # print("current increase: ", increase)

    finallogLikesum = scipy.special.logsumexp(a=logLikelihoods)
    logZ_current = -np.log(nlive) + finallogLikesum + logX_current
    logZ_array = np.array([logZ_previous, logZ_current])
    logZ_total = scipy.special.logsumexp(logZ_array, axis=0)
    # print(samples)

    return {"mean": np.mean(logZ_total),
            "std": np.std(logZ_total)}


logZ = nested_sampling(logLikelihood=logLikelihood, prior=prior, ndim=2, nlive=1000, stop_criterion=1e-3,
                       sampler=rejection_sampler)
print(logZ)
C = 2
sigma = 0.2

# Z = np.math.factorial(C/2)*(2*sigma**2)**(C/2)
# print(np.log(Z))
