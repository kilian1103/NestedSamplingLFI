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
    # random_directions = np.random.normal(silogze=(C,n_samplles))
    # norm = np.linalg.norm(random_directions, axis=0)
    # random_directions/=norm
    return np.random.uniform(low=0, high=1, size=(nsamples, ndim))


def rejection_sampler(ndim, prior, logLikelihood, minlogLike) -> np.ndarray:
    while True:
        proposal_sample = prior(ndim, 1)[0]
        if logLikelihood(proposal_sample, ndim) > minlogLike:
            break
    return proposal_sample


def metropolis_sampler(ndim, samples, logLikelihood, minlogLike, nrepeat=5) -> np.ndarray:
    cov = np.cov(np.array(samples).T)
    random_index = np.random.randint(0, len(samples))
    current_sample = samples[random_index]
    for i in range(nrepeat * ndim):
        while True:
            proposal_sample = multivariate_normal.rvs(mean=current_sample, cov=cov)
            withinPrior = np.logical_and(proposal_sample > 0, proposal_sample < 1).all()
            withinContour = logLikelihood(proposal_sample, ndim) > minlogLike
            if withinPrior and withinContour:
                break
        current_sample = proposal_sample
    return current_sample


def nested_sampling(logLikelihood, prior, ndim, nlive, stop_criterion, sampler):
    # initialisation
    logZ_previous = -1e300 * np.ones(nlive)  # Z = 0
    logX_previous = np.zeros(nlive)  # X = 1
    iteration = 0
    logIncrease = 10  # evidence increase factor

    # sample from prior
    print(f"Sampling {nlive} livepoints from the prior!")
    livepoints = prior(ndim, nlive)
    logLikelihoods = logLikelihood(livepoints, ndim)
    livepoints = livepoints.tolist()
    logLikelihoods = logLikelihoods.tolist()

    while logIncrease > np.log(stop_criterion):
        iteration += 1
        minlogLike = min(logLikelihoods)
        index = logLikelihoods.index(minlogLike)

        # sample t's
        ti_s = np.amax(np.random.random(size=(nlive, nlive)), axis=0)
        log_ti_s = np.log(ti_s)

        logX_current = logX_previous + log_ti_s

        subtraction_coeff = np.array([1, -1]).reshape(2, 1)
        logWeights = np.array([logX_previous, logX_current])
        logWeight_current = scipy.special.logsumexp(a=logWeights, b=subtraction_coeff, axis=0)
        logX_previous = logX_current

        logZ_current = logWeight_current + minlogLike
        logZ_array = np.array([logZ_previous, logZ_current])
        logZ_total = scipy.special.logsumexp(logZ_array, axis=0)
        logZ_previous = logZ_total

        proposal_sample = sampler(ndim, samples=livepoints, logLikelihood=logLikelihood, minlogLike=minlogLike)
        livepoints[index] = proposal_sample.tolist()
        logLikelihoods[index] = float(logLikelihood(proposal_sample, ndim))

        maxlogLike = max(logLikelihoods)
        logIncrease_array = logWeight_current + maxlogLike - logZ_total
        logIncrease = min(logIncrease_array)
        if iteration % 500 == 0:
            print("current iteration: ", iteration)
            # print("current increase: ", increase)

    finallogLikesum = scipy.special.logsumexp(a=logLikelihoods)
    logZ_current = -np.log(nlive) + finallogLikesum + logX_current
    logZ_array = np.array([logZ_previous, logZ_current])
    logZ_total = scipy.special.logsumexp(logZ_array, axis=0)
    print(f"Algorithm terminated after {iteration} iterations!")
    return {"log Z mean": np.mean(logZ_total),
            "log Z std": np.std(logZ_total)}


logZ = nested_sampling(logLikelihood=logLikelihood, prior=prior, ndim=2, nlive=1000, stop_criterion=1e-3,
                       sampler=metropolis_sampler)
print(logZ)
C = 2
sigma = 0.2

# Z = np.math.factorial(C/2)*(2*sigma**2)**(C/2)
# print(np.log(Z))
