import numpy as np
from scipy.stats import multivariate_normal


class Sampler:

    def __init__(self, prior, priorLimits, logLikelihood, ndim):
        self.prior = prior
        self.priorLimits = priorLimits
        self.logLikelihood = logLikelihood
        self.ndim = ndim
        self.samplers = {"Metropolis": Metropolis,
                         "Rejection": Rejection,
                         "MetropolisNRE": MetropolisNRE}

    def getSampler(self, type):
        return self.samplers[type](prior=self.prior, priorLimits=self.priorLimits, logLikelihood=self.logLikelihood,
                                   ndim=self.ndim)


class Metropolis(Sampler):
    def __init__(self, prior, priorLimits, logLikelihood, ndim, **kwargs):
        super().__init__(prior=prior, priorLimits=priorLimits, logLikelihood=logLikelihood, ndim=ndim)

    def sample(self, minlogLike, livepoints, nrepeat=5) -> np.ndarray:
        cov = np.cov(livepoints.T)
        random_index = np.random.randint(0, len(livepoints))
        current_sample = livepoints[random_index].copy()
        lower = self.priorLimits["lower"]
        upper = self.priorLimits["upper"]
        for i in range(nrepeat * self.ndim):
            proposal_sample = multivariate_normal.rvs(mean=current_sample, cov=cov)
            withinPrior = np.logical_and(proposal_sample > lower,
                                         proposal_sample < upper).all()
            withinContour = self.logLikelihood(proposal_sample, self.ndim) > minlogLike
            if withinPrior and withinContour:
                current_sample = proposal_sample.copy()
        return current_sample


class Rejection(Sampler):
    def __init__(self, prior, priorLimits, logLikelihood, ndim):
        super().__init__(prior=prior, priorLimits=priorLimits, logLikelihood=logLikelihood, ndim=ndim)

    def sample(self, minlogLike, **kwargs) -> np.ndarray:
        while True:
            proposal_sample = self.prior(self.ndim, 1)[0]
            if self.logLikelihood(proposal_sample, self.ndim) > minlogLike:
                break
        return proposal_sample


class MetropolisNRE(Sampler):
    def __init__(self, prior, priorLimits, logLikelihood, ndim):
        super().__init__(prior=prior, priorLimits=priorLimits, logLikelihood=logLikelihood, ndim=ndim)

    def sample(self, minlogLike, livepoints, x_0, marginal_indices_3d, nrepeat=5) -> np.ndarray:
        cov = np.cov(livepoints.T)
        random_index = np.random.randint(0, len(livepoints))
        current_sample = livepoints[random_index].copy()
        lower = self.priorLimits["lower"]
        upper = self.priorLimits["upper"]
        for i in range(nrepeat * self.ndim):
            proposal_sample = multivariate_normal.rvs(mean=current_sample, cov=cov)
            withinPrior = np.logical_and(proposal_sample > lower, proposal_sample < upper).all()
            withinContour = self.logLikelihood.log_ratio(observation=x_0, v=[proposal_sample])[
                                marginal_indices_3d].copy() > minlogLike
            if withinPrior and withinContour:
                current_sample = proposal_sample.copy()
        return current_sample
