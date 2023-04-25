from typing import Any, Dict

import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch.distributions import Uniform

from NSLFI.MCMCSampler import Sampler


class NestedSampler:
    def __init__(self, logLikelihood: Any, prior: Dict[str, Uniform], livepoints: Tensor, root="."):
        """
        :param logLikelihood: loglikelihood function given parameters for obs x
        :param prior: uniform prior distribution for parameters
        :param livepoints: list of livepoints
        """
        self.prior = prior
        self.logLikelihood = logLikelihood
        self.livepoints = livepoints
        self.root = root

    def nested_sampling(self, nsim: int, stop_criterion: float, samplertype: str) -> Dict[str, float]:
        """
        :param nsim: number of parallel NS contractions
        :param stop_criterion: evidence stopping criterion
        :param root: root file directory to store results
        :return:
        """
        # standard NS run: 1 sample in 1 sample out
        # initialisation
        logZ_previous = -torch.inf * torch.ones(nsim)  # Z = 0
        logX_previous = torch.zeros(nsim)  # X = 1
        iteration = 0
        logIncrease = 10  # evidence increase factor
        nlive = torch.as_tensor(self.livepoints.shape[0])
        cov = torch.cov(self.livepoints.T)
        cholesky = torch.linalg.cholesky(cov)

        logLikelihoods = self.logLikelihood(self.livepoints)
        livepoints_birthlogL = -torch.inf * torch.ones(nlive)  # L_birth = 0

        # dynamic storage -> lists
        deadpoints = []
        deadpoints_logL = []
        deadpoints_birthlogL = []
        weights = []

        sampler = Sampler(prior=self.prior, logLikelihood=self.logLikelihood).getSampler(
            samplertype)
        while logIncrease > torch.log(torch.as_tensor(stop_criterion)):
            iteration += 1
            # identifying lowest likelihood point
            minlogLike = logLikelihoods.min()
            index = logLikelihoods.argmin()

            # save deadpoint and its loglike
            deadpoint = self.livepoints[index].clone()
            deadpoints.append(deadpoint)
            deadpoints_logL.append(minlogLike)
            deadpoints_birthlogL.append(livepoints_birthlogL[index].clone())

            # sample t's
            ti_s = torch.as_tensor(np.random.power(a=nlive, size=nsim))
            log_ti_s = torch.log(ti_s)

            # Calculate X contraction and weight
            logX_current = logX_previous + log_ti_s
            subtraction_coeff = torch.tensor([1, -1]).reshape(2, 1)
            logWeights = torch.stack([logX_previous, logX_current])
            logWeight_current = torch.as_tensor(scipy.special.logsumexp(a=logWeights, b=subtraction_coeff, axis=0))
            logX_previous = logX_current.clone()
            weights.append(torch.mean(logWeight_current))

            # Calculate evidence increase
            logZ_current = logWeight_current + minlogLike
            logZ_array = torch.stack([logZ_previous, logZ_current])
            logZ_total = torch.logsumexp(logZ_array, axis=0)
            logZ_previous = logZ_total.clone()
            # recompute cov of livepoints
            if iteration % nlive == 0:
                cov = torch.cov(self.livepoints.T)
                cholesky = torch.linalg.cholesky(cov)
            # find new sample satisfying likelihood constraint
            proposal_samples = sampler.sample(livepoints=self.livepoints.clone(), minlogLike=minlogLike,
                                              livelikes=logLikelihoods, cov=cov, cholesky=cholesky,
                                              keep_chain=False)
            proposal_sample, logLike = proposal_samples.pop()

            # replace lowest likelihood sample with proposal sample
            self.livepoints[index] = proposal_sample.clone()
            logLikelihoods[index] = logLike
            livepoints_birthlogL[index] = minlogLike

            maxlogLike = logLikelihoods.max()
            logIncrease_array = logWeight_current + maxlogLike - logZ_total
            logIncrease = logIncrease_array.max()
            if iteration % 500 == 0:
                print("Current log evidence ", logZ_total.max())
                print("current iteration: ", iteration)

        # final <L>*dX sum calculation
        finallogLikesum = torch.logsumexp(logLikelihoods, axis=0)
        logZ_current = -torch.log(nlive) + finallogLikesum + logX_current
        logZ_array = torch.stack([logZ_previous, logZ_current])
        logZ_total = torch.as_tensor(scipy.special.logsumexp(logZ_array, axis=0))

        # convert surviving livepoints to deadpoints
        samples = list(zip(self.livepoints, logLikelihoods, livepoints_birthlogL))
        samples.sort(key=lambda x: x[1], reverse=True)  # sort after logL
        mean_last_shell_weight = torch.mean(logX_current) - torch.log(nlive)
        while len(samples) > 0:
            deadpoint, logLikelihood, deadpoint_birthlogL = samples.pop()
            deadpoints.append(deadpoint)
            deadpoints_logL.append(logLikelihood)
            deadpoints_birthlogL.append(deadpoint_birthlogL)
            weights.append(mean_last_shell_weight)
        torch.save(f=f"{self.root}/weights", obj=torch.stack(weights))
        torch.save(f=f"{self.root}/posterior_samples", obj=torch.stack(deadpoints))
        torch.save(f=f"{self.root}/logL", obj=torch.stack(deadpoints_logL))
        torch.save(f=f"{self.root}/logL_birth", obj=torch.stack(deadpoints_birthlogL))
        print(f"Algorithm terminated after {iteration} iterations!")
        return {"log Z mean": float(torch.mean(logZ_total)),
                "log Z std": float(torch.std(logZ_total))}
