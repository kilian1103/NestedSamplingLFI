from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
from scipy.stats import multivariate_normal, uniform, special_ortho_group


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

    def sample(self, minlogLike, livepoints, livelikes, cov, nrepeat=5, keep_chain=False, **kwargs) -> List[np.ndarray]:
        random_index = np.random.randint(low=0, high=len(livepoints))
        current_sample = livepoints[random_index].copy()
        lower = np.zeros(self.ndim)
        upper = np.zeros(self.ndim)
        chain = []
        for i, val in enumerate(self.prior.values()):
            low, up = val.support()
            lower[i] = low
            upper[i] = up

        for i in range(nrepeat * self.ndim):
            proposal_sample = multivariate_normal.rvs(mean=current_sample, cov=cov)
            withinPrior = np.logical_and(np.greater(proposal_sample, lower), np.less(proposal_sample, upper)).all()
            withinContour = self.logLikelihood(proposal_sample) > minlogLike
            if withinPrior and withinContour:
                if keep_chain:
                    chain.append(proposal_sample)
                current_sample = proposal_sample.copy()
        if keep_chain:
            return chain
        else:
            return [current_sample]


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
        return [proposal_sample]


class Slice(Sampler):
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike, livepoints, livelikes, cov, cholesky, nrepeat=5, step_size=0.1,
               keep_chain=False) -> List[np.ndarray]:
        # uniform prior bounds
        lower = np.zeros(self.ndim)
        upper = np.zeros(self.ndim)
        for i, val in enumerate(self.prior.values()):
            low, up = val.support()
            lower[i] = low
            upper[i] = up
        # choose randomly existing livepoint satisfying likelihood constraint
        random_index = np.random.randint(low=0, high=len(livepoints))
        current_sample = livepoints[random_index].copy()
        chain = []

        # get random orthonormal basis to slice on
        ortho_norm = special_ortho_group.rvs(dim=self.ndim)
        x_l, x_r, idx = self._extend_nd_interval(current_sample=current_sample, step_size=step_size,
                                                 minlogLike=minlogLike, ortho_norm=ortho_norm, cholesky=cholesky)

        for i in range(nrepeat * self.ndim):
            # sample along slice
            u = np.random.uniform(low=0, high=1)
            intermediate_sample = u * x_l + (1 - u) * x_r

            withinPrior = np.logical_and(np.greater(intermediate_sample, lower),
                                         np.less(intermediate_sample, upper)).all()
            withinContour = self.logLikelihood(intermediate_sample) > minlogLike
            if withinPrior and withinContour:
                # accept sample
                if keep_chain:
                    chain.append(intermediate_sample)
                current_sample = intermediate_sample.copy()
                # slice along new n_vector
                x_l, x_r, idx = self._extend_nd_interval(current_sample=current_sample, step_size=step_size,
                                                         minlogLike=minlogLike, ortho_norm=ortho_norm,
                                                         cholesky=cholesky)
            else:
                # rescale bounds if point is not within contour or prior
                dist_proposal = np.linalg.norm(x_l - intermediate_sample)
                dist_origin = np.linalg.norm(x_l - current_sample)
                if dist_proposal > dist_origin:
                    x_r = intermediate_sample
                else:
                    x_l = intermediate_sample
        if keep_chain:
            return chain
        else:
            return [current_sample]

    def _extend_1d_interval(self, current_sample, step_size, minlogLike):
        # chose random coordinate axis
        randIdx = np.random.randint(low=0, high=self.ndim)
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

    def _extend_nd_interval(self, current_sample, step_size, minlogLike, ortho_norm, cholesky):
        # chose random orthonorm axis
        randIdx = np.random.randint(low=0, high=self.ndim)
        n_vec = ortho_norm[randIdx]
        n_dir = np.matmul(cholesky, n_vec)
        x_l = current_sample.copy()
        x_r = current_sample.copy()
        # extend bounds along slice
        r = np.random.uniform(0, 1)
        x_l -= r * step_size * n_dir
        x_r += (1 - r) * step_size * n_dir

        while self.logLikelihood(x_l) > minlogLike:
            x_l -= step_size * n_dir
        while self.logLikelihood(x_r) > minlogLike:
            x_r += step_size * n_dir
        return x_l, x_r, randIdx
