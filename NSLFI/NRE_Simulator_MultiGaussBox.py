import scipy.stats as stats
import swyft
import torch

from NSLFI.NRE_Settings import NRE_Settings


class Simulator(swyft.Simulator):
    def __init__(self, nreSettings: NRE_Settings):
        super().__init__()
        self.nreSettings = nreSettings
        self.n = self.nreSettings.num_features
        self.d = self.nreSettings.num_features_dataset
        self.m = torch.zeros(self.d)  # mean vec of dataset
        self.M = torch.eye(n=self.d, m=self.n)  # transform matrix of dataset to parameter vee
        self.C = torch.eye(self.d)  # cov matrix of dataset
        self.mu = torch.zeros(self.n)  # mean vec of parameter prior
        self.Sigma = torch.eye(self.n)  # cov matrix of parameter prior
        self.Sigma_inv = torch.inverse(self.Sigma)
        self.C_inv = torch.inverse(self.C)

    def xgivenz(self, z):
        """likelihood"""
        return stats.multivariate_normal(mean=(self.m + self.M @ z), cov=self.C).rvs()

    def logratio(self, x, z):
        pass

    def zgivenx(self, z, x):
        """posterior"""
        pass

    def build(self, graph):
        z = graph.node(self.nreSettings.targetKey, lambda t: stats.uniform(loc=self.nreSettings.sim_prior_lower,
                                                                           scale=self.nreSettings.prior_width).rvs())
        x = graph.node(self.nreSettings.obsKey, self.xgivenz, z)
