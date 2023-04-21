from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from scipy.stats import special_ortho_group
from torch import Tensor
from torch.distributions import MultivariateNormal, Uniform


class Sampler:
    def __init__(self, prior: Dict[str, Uniform], logLikelihood: Any):
        self.prior = prior
        self.logLikelihood = logLikelihood
        self.ndim = len(prior)
        self._samplers = {"Metropolis": Metropolis,
                          "Rejection": Rejection,
                          "Slice": Slice}
        self.lower = torch.tensor([val.low for val in self.prior.values()])
        self.upper = torch.tensor([val.high for val in self.prior.values()])

    def getSampler(self, type: str):
        return self._samplers[type](prior=self.prior, logLikelihood=self.logLikelihood)

    @abstractmethod
    def sample(self, **kwargs) -> List[Tuple[Tensor, Tensor]]:
        raise NotImplementedError("This is an abstract method, please implement an appropriate sampling class")


class Metropolis(Sampler):
    def __init__(self, prior: Dict[str, Uniform], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike: Tensor, livepoints: Tensor, livelikes: Tensor, cov: Tensor,
               nrepeat=5, keep_chain=False, **kwargs) -> List[Tuple[Tensor, Tensor]]:
        random_index = torch.randint(low=0, high=len(livepoints), size=(1,))
        current_sample = livepoints[random_index].clone()
        logLike = livelikes[random_index].clone()
        chain = []
        for i in range(nrepeat * self.ndim):
            proposal_sample = MultivariateNormal(loc=current_sample, covariance_matrix=cov).sample()
            withinPrior = torch.logical_and(torch.greater(proposal_sample, self.lower),
                                            torch.less(proposal_sample, self.upper)).all()
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
    def __init__(self, prior: Dict[str, Uniform], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike: Tensor, **kwargs) -> List[Tuple[Tensor, Tensor]]:
        while True:
            proposal_sample = Uniform(low=0, high=1).sample(sample_shape=(self.ndim,))
            logLike_prop = self.logLikelihood(proposal_sample)
            if logLike_prop > minlogLike:
                break
        return [(proposal_sample, logLike_prop)]


class Slice(Sampler):
    def __init__(self, prior: Dict[str, Uniform], logLikelihood: Any):
        super().__init__(prior=prior, logLikelihood=logLikelihood)

    def sample(self, minlogLike: Tensor, livepoints: Tensor, livelikes: Tensor,
               cholesky: Tensor, nrepeat=5, step_size=2,
               keep_chain=False, **kwargs) -> List[Tuple[Tensor, Tensor]]:

        chain = []  # list of accepted samples
        # choose randomly existing livepoint satisfying likelihood constraint
        random_index = torch.randint(low=0, high=len(livepoints), size=(1,))
        current_sample = livepoints[random_index].clone()
        logLike = livelikes[random_index].clone()

        # get random orthonormal basis to slice on
        ortho_norm = torch.as_tensor(special_ortho_group.rvs(dim=self.ndim, size=nrepeat)).reshape(nrepeat * self.ndim,
                                                                                                   self.ndim)
        accepted = True  # boolean to track state of sampling
        x_l, x_r = None, None  # initialization of bounds
        num_accepted = 0  # number of accepted slice samples

        while num_accepted < nrepeat * self.ndim:
            if accepted:
                # slice along new n_vector
                x_l, x_r = self._extend_nd_interval(current_sample=current_sample, step_size=step_size,
                                                    minlogLike=minlogLike, cholesky=cholesky,
                                                    n_vec=ortho_norm[num_accepted])
                num_accepted += 1

            # sample along slice
            u = torch.rand(1)
            proposal_sample = u * x_l + (1 - u) * x_r

            withinPrior = torch.logical_and(torch.greater(proposal_sample, self.lower),
                                            torch.less(proposal_sample, self.upper)).all()
            logLike_prop = self.logLikelihood(proposal_sample)
            withinContour = logLike_prop > minlogLike
            if withinPrior and withinContour:
                # accept sample
                accepted = True
                if keep_chain:
                    chain.append((proposal_sample, logLike_prop))
                current_sample = proposal_sample.clone()
                logLike = logLike_prop.clone()
            else:
                # rescale bounds if point is not within contour or prior
                # dot product for angle check between vectors
                accepted = False
                if torch.dot((proposal_sample - current_sample).squeeze(), (x_r - current_sample).squeeze()) > 0:
                    x_r = proposal_sample.clone()
                else:
                    x_l = proposal_sample.clone()
        if keep_chain:
            return chain
        else:
            return [(current_sample, logLike)]

    def _extend_nd_interval(self, current_sample: Tensor, step_size: float, minlogLike: Tensor, n_vec: Tensor,
                            cholesky: Tensor) -> Tuple[Tensor, Tensor]:
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
        return x_l, x_r
