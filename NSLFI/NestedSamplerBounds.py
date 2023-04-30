from typing import Any, Dict

import torch
from torch import Tensor
from torch.distributions import Uniform

from NSLFI.NestedSampler import NestedSampler


class NestedSamplerBounds(NestedSampler):
    def __init__(self, logLikelihood: Any, prior: Dict[str, Uniform], livepoints: Tensor, samplertype: str, root="."):
        super().__init__(logLikelihood=logLikelihood, prior=prior, livepoints=livepoints, samplertype=samplertype,
                         root=root)

    def nested_sampling(self, stop_criterion: float, boundarySample: Tensor, nsamples=2000, keep_chain=True) -> Dict[
        str, float]:

        # NS run with rounds, and constant median Likelihood constraint for each round

        logLikelihoods = self.logLikelihood(self.livepoints)

        # dynamic storage -> lists
        deadpoints = []
        deadpoints_logL = []
        deadpoints_birthlogL = []

        # define truncation boundary criterion
        boundarySampleLogLike = self.logLikelihood(boundarySample)

        self.livepoints = self.livepoints[logLikelihoods > boundarySampleLogLike]
        logLikelihoods = logLikelihoods[logLikelihoods > boundarySampleLogLike]
        cov = torch.cov(self.livepoints.T)
        cholesky = torch.linalg.cholesky(cov)

        while len(deadpoints) < nsamples:
            # find new samples satisfying likelihood constraint
            proposal_samples = self.sampler.sample(livepoints=self.livepoints.clone(),
                                                   minlogLike=boundarySampleLogLike,
                                                   livelikes=logLikelihoods, cov=cov, cholesky=cholesky,
                                                   keep_chain=keep_chain)
            # add new samples to deadpoints
            while len(proposal_samples) > 0:
                proposal_sample, logLike = proposal_samples.pop()
                deadpoints.append(proposal_sample)
                deadpoints_birthlogL.append(boundarySampleLogLike)
                deadpoints_logL.append(logLike)
                if len(deadpoints) == nsamples:
                    break
        torch.save(f=f"{self.root}/posterior_samples", obj=torch.stack(deadpoints).squeeze())
        torch.save(f=f"{self.root}/logL", obj=torch.as_tensor(deadpoints_logL))
        torch.save(f=f"{self.root}/logL_birth", obj=torch.as_tensor(deadpoints_birthlogL))
        return {"log Z mean": 0, "log Z std": 0}
