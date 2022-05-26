import numpy as np
from scipy.stats import multivariate_normal


class Sampler:

    def __init__(self, prior, logLikelihood, ndim):
        self.prior = prior
        self.logLikelihood = logLikelihood
        self.ndim = ndim
        self.samplers = {"Metropolis": Metropolis,
                         "Rejection": Rejection}

    def getSampler(self, type):
        return self.samplers[type](prior=self.prior, logLikelihood=self.logLikelihood, ndim=self.ndim)


class Metropolis(Sampler):
    def __init__(self, prior, logLikelihood, ndim):
        super().__init__(prior=prior, logLikelihood=logLikelihood, ndim=ndim)

    def sample(self, minlogLike, livepoints, nrepeat=5) -> np.ndarray:
        cov = np.cov(np.array(livepoints).T)
        random_index = np.random.randint(0, len(livepoints))
        current_sample = livepoints[random_index].copy()
        for i in range(nrepeat * self.ndim):
            while True:
                proposal_sample = multivariate_normal.rvs(mean=current_sample, cov=cov)
                withinPrior = np.logical_and(proposal_sample > 0, proposal_sample < 1).all()
                withinContour = self.logLikelihood(proposal_sample, self.ndim) > minlogLike
                if withinPrior and withinContour:
                    break
            current_sample = proposal_sample
        return current_sample


class Rejection(Sampler):
    def __init__(self, prior, logLikelihood, ndim):
        super().__init__(prior=prior, logLikelihood=logLikelihood, ndim=ndim)

    def sample(self, minlogLike, **kwargs) -> np.ndarray:
        while True:
            proposal_sample = self.prior(self.ndim, 1)[0]
            if self.logLikelihood(proposal_sample, self.ndim) > minlogLike:
                break
        return proposal_sample
