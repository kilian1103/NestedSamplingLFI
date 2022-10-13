import numpy as np
from margarine.maf import MAF
from pypolychord.priors import UniformPrior


class MAF_Poly:
    """Polychord wrapper"""

    def __init__(self, maf: MAF):
        self.MAF = maf
        self.nDims = 2
        self.nDerived = 0

    def prior(self, cube):
        """Prior for Gaussian data Likelihood.

        This function is used for the standard run.
        """
        theta = np.zeros_like(cube)
        theta[-2] = UniformPrior(0, 1)(
            cube[-2])  #
        theta[-1] = UniformPrior(0, 1)(
            cube[-1])  #
        return theta

    def loglike(self, theta):
        proposal_sample = np.array([theta[0], theta[1]])
        loglikelihood = self.MAF.log_prob(params=proposal_sample)
        r2 = 0
        return loglikelihood, [r2]
