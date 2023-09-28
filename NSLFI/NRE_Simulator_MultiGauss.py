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
        self.z_sampler = stats.multivariate_normal(mean=self.mu, cov=self.Sigma).rvs

    def xgivenz(self, z):
        return stats.multivariate_normal(mean=(self.m + self.M @ z), cov=self.C).rvs()

    def logratio(self, x, z):
        loglike = stats.multivariate_normal(mean=(self.m + self.M @ z), cov=self.C).logpdf(x)
        logevidence = stats.multivariate_normal(mean=(self.m + self.M @ self.mu),
                                                cov=(self.C + self.M @ self.Sigma @ self.M.T)).logpdf(x)
        logratio = (loglike - logevidence).sum()
        return logratio

    def zgivenx(self, x):
        """Posterior sampling"""
        return stats.multivariate_normal(mean=(self.Sigma_inv + self.M.T @ self.C_inv @ self.M).inverse() @ (
                self.Sigma_inv @ self.mu + self.M.T @ self.C_inv @ (torch.as_tensor(x).float() - self.m)),
                                         cov=(
                                                 self.Sigma_inv + self.M.T @ self.C_inv @
                                                 self.M).inverse()).rvs()

    def build(self, graph):
        z = graph.node(self.nreSettings.targetKey, self.z_sampler)
        x = graph.node(self.nreSettings.obsKey, self.xgivenz, z)
        l = graph.node(self.nreSettings.contourKey, self.logratio, x, z)
        post = graph.node(self.nreSettings.posteriorsKey, self.zgivenx, x)
