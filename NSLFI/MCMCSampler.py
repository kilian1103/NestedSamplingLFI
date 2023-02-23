from abc import abstractmethod
from typing import Any, Dict

import numpy as np
from scipy.stats import multivariate_normal, uniform


class Sampler:

    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        self.prior = prior
        self.logLikelihood = logLikelihood
        self.ndim = len(prior)
        self.samplers = {"Metropolis": Metropolis,
                         "Rejection": Rejection,
                         "Slice": Slice}

    def getSampler(self, type):
        return self.samplers[type](prior=self.prior, logLikelihood=self.logLikelihood)

    @abstractmethod
    def sample(self, **kwargs) -> np.ndarray:
        raise NotImplementedError("This is an abstract method, please implement an appropriate sampling class")


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


class Slice(Sampler):
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike, livepoints, livelikes, nrepeat=5, step_size=0.1) -> np.ndarray:
        # uniform prior bounds
        lower = np.zeros(self.ndim)
        upper = np.zeros(self.ndim)
        for i, val in enumerate(self.prior.values()):
            low, up = val.support()
            lower[i] = low
            upper[i] = up
        # choose randomly existing livepoint satisfying likelihood constraint
        indices = np.arange(len(livelikes))[livelikes > minlogLike]
        random_index = np.random.choice(indices)
        current_sample = livepoints[random_index].copy()

        # define temp sample to slice on
        intermediate_sample = current_sample.copy()
        x_l, x_r, idx = self._extend_1d_interval(intermediate_sample, step_size, minlogLike)
        for i in range(nrepeat * self.ndim):
            x_n_new = np.random.uniform(x_l[idx], x_r[idx])
            intermediate_sample[idx] = x_n_new
            withinPrior = np.logical_and(np.greater(intermediate_sample, lower),
                                         np.less(intermediate_sample, upper)).all()
            withinContour = self.logLikelihood(intermediate_sample) > minlogLike
            if withinPrior and withinContour:
                # accept sample
                current_sample = intermediate_sample.copy()
                # slice along new axis
                x_l, x_r, idx = self._extend_1d_interval(current_sample, step_size, minlogLike)
            else:
                # rescale bounds if point is not within contour or prior
                if intermediate_sample[idx] > current_sample[idx]:
                    x_r[idx] = intermediate_sample[idx]
                else:
                    x_l[idx] = intermediate_sample[idx]

        return current_sample

    def _extend_1d_interval(self, current_sample, step_size, minlogLike):
        # chose random coordinate axis
        randIdx = np.random.choice(np.arange(self.ndim))
        x_l = current_sample.copy()
        x_r = current_sample.copy()
        r = np.random.uniform(0, 1)
        # and slice along its axis
        x_l[randIdx] -= r * step_size
        x_r[randIdx] += (1 - r) * step_size

        while self.logLikelihood(x_l) > minlogLike:
            x_l[randIdx] -= step_size
        while self.logLikelihood(x_r) > minlogLike:
            x_r[randIdx] += step_size
        return x_l, x_r, randIdx
