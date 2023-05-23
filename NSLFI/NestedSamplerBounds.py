from typing import Any, Dict

import torch
from mpi4py import MPI
from torch import Tensor
from torch.distributions import Uniform

from NSLFI.NestedSampler import NestedSampler


class NestedSamplerBounds(NestedSampler):
    def __init__(self, logLikelihood: Any, prior: Dict[str, Uniform], livepoints: Tensor, samplertype: str, root=".",
                 logLs=None):
        super().__init__(logLikelihood=logLikelihood, prior=prior, livepoints=livepoints, samplertype=samplertype,
                         root=root, logLs=logLs)

    def nested_sampling(self, stop_criterion: float, boundarySample: Tensor, nsamples=2000, keep_chain=True) -> Dict[
        str, float]:
        """A Nested Sampler that uses a truncation boundary to sample from the likelihood.
        :param stop_criterion: placeholder: float
        :param boundarySample: sample that defines the truncation boundary: Tensor
        :param nsamples: number of samples to be drawn: int
        :param keep_chain: keep the chain of samples that led to the final sample: True/False
        :return:
        """
        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()
        # dynamic storage -> lists
        deadpoints = []
        deadpoints_logL = []
        deadpoints_birthlogL = []

        # define truncation boundary criterion
        boundarySampleLogLike = self.logLikelihood(boundarySample)

        self.livepoints = self.livepoints[self.logLikelihoods > boundarySampleLogLike]
        self.logLikelihoods = self.logLikelihoods[self.logLikelihoods > boundarySampleLogLike]
        cov = torch.cov(self.livepoints.T)
        cholesky = torch.linalg.cholesky(cov)
        # prepare for MPI
        nsamples_per_core = nsamples // size_gen
        if rank_gen == 0:
            # add remainder to first core
            nsamples_per_core += nsamples % size_gen
        while len(deadpoints) < nsamples_per_core:
            # find new samples satisfying likelihood constraint
            proposal_samples = self.sampler.sample(livepoints=self.livepoints.clone(),
                                                   minlogLike=boundarySampleLogLike,
                                                   livelikes=self.logLikelihoods, cov=cov, cholesky=cholesky,
                                                   keep_chain=keep_chain)
            # add new samples to deadpoints
            while len(proposal_samples) > 0:
                proposal_sample, logLike = proposal_samples.pop()
                deadpoints.append(proposal_sample)
                deadpoints_birthlogL.append(boundarySampleLogLike)
                deadpoints_logL.append(logLike)
                if len(deadpoints) == nsamples_per_core:
                    break

        comm_gen.Barrier()
        deadpoints = comm_gen.gather(deadpoints, root=0)
        deadpoints_logL = comm_gen.gather(deadpoints_logL, root=0)
        deadpoints_birthlogL = comm_gen.gather(deadpoints_birthlogL, root=0)
        comm_gen.Barrier()
        if rank_gen == 0:
            deadpoints = [item for sublist in deadpoints for item in sublist]
            deadpoints_logL = [item for sublist in deadpoints_logL for item in sublist]
            deadpoints_birthlogL = [item for sublist in deadpoints_birthlogL for item in sublist]
            torch.save(f=f"{self.root}/posterior_samples", obj=torch.stack(deadpoints).squeeze())
            torch.save(f=f"{self.root}/logL", obj=torch.as_tensor(deadpoints_logL))
            torch.save(f=f"{self.root}/logL_birth", obj=torch.as_tensor(deadpoints_birthlogL))
        return {"log Z mean": 0, "log Z std": 0}
