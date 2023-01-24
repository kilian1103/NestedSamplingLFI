from typing import Any, Dict

import numpy as np
from scipy.stats import multivariate_normal, uniform


class Sampler:

    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        self.prior = prior
        self.logLikelihood = logLikelihood
        self.ndim = len(prior)
        self.samplers = {"Metropolis": Metropolis,
                         "Rejection": Rejection}

    def getSampler(self, type):
        return self.samplers[type](prior=self.prior, logLikelihood=self.logLikelihood)


class Metropolis(Sampler):
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any, ):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike, livepoints, livelikes, nrepeat=5) -> np.ndarray:
        cov = np.cov(livepoints.T)
        i = np.arange(len(livelikes))[livelikes > minlogLike]
        random_index = np.random.choice(i)
        current_sample = livepoints[random_index].copy()
        lower = np.zeros(self.ndim)
        upper = np.zeros(self.ndim)
        for i, val in enumerate(self.prior.values()):
            low, up = val.support()
            lower[i] = low
            upper[i] = up

        for i in range(nrepeat * self.ndim):
            proposal_sample = multivariate_normal.rvs(mean=current_sample, cov=cov)
            withinPrior = np.logical_and(np.greater(proposal_sample, lower), np.less(proposal_sample, upper)).all()
            withinContour = self.logLikelihood(proposal_sample) > minlogLike
            if withinPrior and withinContour:
                current_sample = proposal_sample.copy()
        return current_sample


class Rejection(Sampler):
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike, **kwargs) -> np.ndarray:
        lower = np.zeros(self.ndim)
        upper = np.zeros(self.ndim)
        for i, val in enumerate(self.prior.values()):
            low, up = val.support()
            lower[i] = low
            upper[i] = up
        while True:
            proposal_sample = uniform.rvs(loc=lower, scale=upper - lower)
            if self.logLikelihood(proposal_sample) > minlogLike:
                break
        return proposal_sample
