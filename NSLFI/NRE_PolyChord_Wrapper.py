import numpy as np
import swyft
from pypolychord.priors import UniformPrior
from scipy.stats import multivariate_normal


class NRE_Poly:
    """Polychord wrapper"""

    def __init__(self, nre: swyft.MarginalRatioEstimator, x_0: np.ndarray):
        self.NRE = nre
        self.x_0 = x_0
        self.nDims = 2
        self.nDerived = 0

    def prior(self, cube):
        """Prior for Gaussian data Likelihood.

        This function is used for the standard run.
        """
        theta = np.zeros_like(cube)
        theta[-2] = UniformPrior(0, 1)(
            cube[-2])  # Phase_sys parameter prior is uniform between 0 and 2pi
        theta[-1] = UniformPrior(0, 1)(
            cube[-1])  # noise has a log uniform prior from 10^-4 to 10^-1
        return theta

    def loglike(self, theta):
        proposal_sample = np.array([theta[0], theta[1]]).tolist()
        loglikelihood = float(self.NRE.log_ratio(observation=self.x_0, v=[proposal_sample])[(0, 1)])
        r2 = 0
        return loglikelihood, [r2]

    def toylogLikelihood(self, theta):
        # Multivariate Gaussian centred at X = 0.5, y= 0.5
        means = np.array([theta[0], theta[1]])
        cov = 0.01 * np.eye(N=self.nDims)
        loglikelihood = multivariate_normal.logpdf(x=self.x_0["x"].T, mean=means, cov=cov).sum()
        r2 = 0
        return loglikelihood, [r2]
