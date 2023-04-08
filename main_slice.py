import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from NSLFI.NestedSampler import nested_sampling


def main():
    torch.random.manual_seed(234)
    np.random.seed(234)
    samplerType = "Slice"
    # samplerType = "Metropolis"
    logZs = []
    logZsErr = []
    maxDim = 20  # number of dimensions to test
    for n in range(2, maxDim + 1):
        # problem parameter
        ndim = n
        nlive = 25 * ndim
        means = 0.5 * torch.ones(ndim)
        cov = 0.01 * torch.eye(ndim)
        mvNormal = MultivariateNormal(loc=means, covariance_matrix=cov)

        def logLikelihood(x) -> torch.tensor:
            return mvNormal.log_prob(x)

        priors = {f"theta_{i}": torch.distributions.uniform.Uniform(low=0, high=1) for i in range(ndim)}
        livepoints = priors["theta_0"].sample(sample_shape=(nlive, ndim)).type(torch.float64)

        logZ = nested_sampling(logLikelihood=logLikelihood, prior=priors,
                               nsim=100, stop_criterion=1e-3, livepoints=livepoints, samplertype=samplerType,
                               keep_chain=False,
                               rounds=0)
        logZs.append(logZ["log Z mean"])
        logZsErr.append(logZ["log Z std"])

    plt.figure()
    plt.plot([x for x in range(2, maxDim + 1)], logZs)
    plt.errorbar([x for x in range(2, maxDim + 1)], logZs, yerr=logZsErr, fmt='o')
    plt.title(f"{samplerType} sampler evidences for Gaussian likelihood with uniform prior")
    plt.xlabel("Dimension")
    plt.ylabel("log Z")
    plt.savefig(f"{samplerType}_gaussian.pdf")

    if __name__ == "__main__":
        main()
