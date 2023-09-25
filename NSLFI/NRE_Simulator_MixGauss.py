import numpy as np
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
        self.a = self.nreSettings.num_mixture_components
        self.a_vec = self.a_sampler()
        self.mu_theta = torch.rand(size=(1, self.n))  # random mean vec of
        # parameter
        self.M = torch.rand(size=(self.a, self.d, self.n))  # random transform matrix of param to data space vec
        self.mu_data = torch.randint(low=self.nreSettings.sim_prior_lower,
                                     high=(self.nreSettings.sim_prior_lower + self.nreSettings.prior_width),
                                     size=(self.a, self.d)).float() + torch.matmul(self.M,
                                                                                   self.mu_theta.unsqueeze(
                                                                                       2)).squeeze(
            2)  # random data mean vec
        self.Sigma = torch.eye(self.n)  # cov matrix of parameter prior
        self.S = torch.eye(self.d) + self.M @ self.Sigma @ torch.transpose(self.M, 1, 2)  # cov matrix of dataset
        self.X = self.M @ self.Sigma  # covariance entries between data and parameter

    def a_sampler(self):
        a_components = np.random.rand(self.a)
        a_components = a_components / a_components.sum()
        return a_components

    def dataGivenA_sampler(self, idx: int):
        return stats.multivariate_normal(mean=self.mu_data[idx].squeeze(), cov=self.S[idx].squeeze()).rvs()

    def thetaGivenA_sampler(self, idx: int):
        # same param prior model
        return stats.multivariate_normal(mean=self.mu_theta.squeeze(), cov=self.Sigma).rvs()

    def thetaGivenDataGivenA(self, D: np.ndarray, idx: int):
        mean = self.mu_theta + (torch.transpose(self.X, 1, 2) @ torch.inverse(self.S) @ (
                torch.as_tensor(D).float() - self.mu_data).unsqueeze(2)).squeeze(2)
        cov = self.Sigma - torch.transpose(self.X, 1, 2) @ self.S @ self.X
        rv = stats.multivariate_normal(mean=mean[idx].squeeze(), cov=cov[idx].squeeze()).rvs()
        return rv

    def dataGivenThetaGivenA(self, theta: np.ndarray, idx: int):
        mean = self.mu_data + (self.X @ torch.inverse(self.Sigma) @ (
                torch.as_tensor(theta).float() - self.mu_theta).unsqueeze(2)).squeeze(2)
        cov = self.S - self.X @ self.Sigma @ torch.transpose(self.X, 1, 2)
        rv = stats.multivariate_normal(mean=mean[idx].squeeze(), cov=cov[idx].squeeze()).rvs()
        return rv

    def logratio(self, x, z):
        loglike = stats.multivariate_normal(mean=(self.m + self.M @ z), cov=self.C).logpdf(x)
        logevidence = stats.multivariate_normal(mean=(self.m + self.M @ self.mu),
                                                cov=(self.C + self.M @ self.Sigma @ self.M.T)).logpdf(x)
        logratio = (loglike - logevidence).sum()
        return logratio

    def build(self, graph):
        i = graph.node("i", np.random.choice, self.a, 1, True, self.a_vec)
        x_i = graph.node("x_i", self.dataGivenA_sampler, i)
        z_i = graph.node("z_i", self.thetaGivenA_sampler, i)
        z = graph.node(self.nreSettings.targetKey, self.thetaGivenDataGivenA, x_i, i)
        x = graph.node(self.nreSettings.obsKey, self.dataGivenThetaGivenA, z_i, i)
        # true answer
        # zgivenX_i = graph.node(self.nreSettings.contourKey, self.logratio, x_i, z, i)
