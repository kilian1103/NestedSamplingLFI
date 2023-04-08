from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from scipy.stats import special_ortho_group
from torch.distributions import MultivariateNormal, Uniform


class Sampler:
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        self.prior = prior
        self.logLikelihood = logLikelihood
        self.ndim = len(prior)
        self.samplers = {"Metropolis": Metropolis,
                         "Rejection": Rejection,
                         "Slice": Slice}

    def getSampler(self, type: str):
        return self.samplers[type](prior=self.prior, logLikelihood=self.logLikelihood)


@abstractmethod
def sample(self, **kwargs) -> List[torch.tensor]:
    raise NotImplementedError("This is an abstract method, please implement an appropriate sampling class")


class Metropolis(Sampler):
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike: torch.tensor, livepoints: torch.tensor, livelikes: torch.tensor, cov: torch.tensor,
               nrepeat=5, keep_chain=False, **kwargs) -> List[Tuple[torch.tensor, torch.tensor]]:
        random_index = torch.randint(low=0, high=len(livepoints), size=(1,))
        current_sample = livepoints[random_index].clone()
        logLike = livelikes[random_index].clone()
        lower = torch.zeros(self.ndim)
        upper = torch.zeros(self.ndim)
        chain = []
        for i, val in enumerate(self.prior.values()):
            low, up = val.low, val.high
            lower[i] = low
            upper[i] = up

        for i in range(nrepeat * self.ndim):
            proposal_sample = MultivariateNormal(loc=current_sample, covariance_matrix=cov).sample()
            withinPrior = torch.logical_and(torch.greater(proposal_sample, lower),
                                            torch.less(proposal_sample, upper)).all()
            logLike_prop = self.logLikelihood(proposal_sample)
            withinContour = logLike_prop > minlogLike
            if withinPrior and withinContour:
                if keep_chain:
                    chain.append((proposal_sample, logLike_prop))
                current_sample = proposal_sample.clone()
                logLike = logLike_prop.clone()
        if keep_chain:
            return chain
        else:
            return [(current_sample, logLike)]


class Rejection(Sampler):
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike: torch.tensor, **kwargs) -> List[Tuple[torch.tensor, torch.tensor]]:
        lower = torch.zeros(self.ndim)
        upper = torch.zeros(self.ndim)
        for i, val in enumerate(self.prior.values()):
            low, up = val.low, val.high
            lower[i] = low
            upper[i] = up
        while True:
            proposal_sample = Uniform(low=0, high=1).sample(sample_shape=(self.ndim,))
            logLike_prop = self.logLikelihood(proposal_sample)
            if logLike_prop > minlogLike:
                break
        return [(proposal_sample, logLike_prop)]


class Slice(Sampler):
    def __init__(self, prior: Dict[str, Any], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike: torch.tensor, livepoints: torch.tensor, livelikes: torch.tensor,
               cholesky: torch.tensor, nrepeat=5, step_size=2,
               keep_chain=False, **kwargs) -> List[Tuple[torch.tensor, torch.tensor]]:

        chain = []  # list of accepted samples
        # set up prior bounds for checking if sample is within prior
        lower = torch.zeros(self.ndim)
        upper = torch.zeros(self.ndim)
        for i, val in enumerate(self.prior.values()):
            low, up = val.low, val.high
            lower[i] = low
            upper[i] = up
        # choose randomly existing livepoint satisfying likelihood constraint
        random_index = torch.randint(low=0, high=len(livepoints), size=(1,))
        current_sample = livepoints[random_index].clone()
        logLike = livelikes[random_index].clone()

        # get random orthonormal basis to slice on
        ortho_norm = torch.tensor(special_ortho_group.rvs(dim=self.ndim))
        x_l, x_r, idx = self._extend_nd_interval(current_sample=current_sample, step_size=step_size,
                                                 minlogLike=minlogLike, ortho_norm=ortho_norm, cholesky=cholesky)

        for i in range(nrepeat * self.ndim):
            # sample along slice
            u = torch.rand(1)
            intermediate_sample = u * x_l + (1 - u) * x_r

            withinPrior = torch.logical_and(torch.greater(intermediate_sample, lower),
                                            torch.less(intermediate_sample, upper)).all()
            logLike_prop = self.logLikelihood(intermediate_sample)
            withinContour = logLike_prop > minlogLike
            if withinPrior and withinContour:
                # accept sample
                if keep_chain:
                    chain.append((intermediate_sample, logLike_prop))
                current_sample = intermediate_sample.clone()
                logLike = logLike_prop.clone()
                # slice along new n_vector
                x_l, x_r, idx = self._extend_nd_interval(current_sample=current_sample, step_size=step_size,
                                                         minlogLike=minlogLike, ortho_norm=ortho_norm,
                                                         cholesky=cholesky)
            else:
                # rescale bounds if point is not within contour or prior
                # dot product for angle check between vectors
                if torch.dot((intermediate_sample - current_sample).squeeze(), (x_r - current_sample).squeeze()) > 0:
                    x_r = intermediate_sample.clone()
                else:
                    x_l = intermediate_sample.clone()
        if keep_chain:
            return chain
        else:
            return [(current_sample, logLike)]

    def _extend_nd_interval(self, current_sample: torch.tensor, step_size: float, minlogLike: torch.tensor,
                            ortho_norm: torch.tensor, cholesky: torch.tensor) -> Tuple[
        torch.tensor, torch.tensor, torch.tensor]:
        # chose random orthonorm axis
        randIdx = torch.randint(low=0, high=self.ndim, size=(1,))
        n_vec = ortho_norm[randIdx].squeeze()
        n_dir = torch.matmul(cholesky, n_vec)
        x_l = current_sample.clone()
        x_r = current_sample.clone()
        # extend bounds along slice
        r = torch.rand(1)
        x_l -= r * step_size * n_dir
        x_r += (1 - r) * step_size * n_dir

        while self.logLikelihood(x_l) > minlogLike:
            x_l -= step_size * n_dir
        while self.logLikelihood(x_r) > minlogLike:
            x_r += step_size * n_dir
        return x_l, x_r, randIdx
