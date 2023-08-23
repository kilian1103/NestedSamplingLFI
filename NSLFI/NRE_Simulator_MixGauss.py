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
        self.mu_theta = torch.rand(size=(self.a, self.n))  # random mean vec of parameter
        self.M = torch.rand(size=(self.a, self.d, self.n))  # random transform matrix of param to data space vec
        self.mu_data = torch.rand(size=(self.a, self.d)) + torch.matmul(self.M, self.mu_theta.unsqueeze(2)).squeeze(
            2)  # random data mean vec
        self.Sigma = torch.eye(self.n)  # cov matrix of parameter prior
        self.S = torch.eye(self.d) + self.M @ self.Sigma @ torch.transpose(self.M, 1, 2)  # cov matrix of dataset
        self.X = self.M @ self.Sigma  # covariance entries between data and parameter

    def a_sampler(self):
        a_components = torch.rand(size=(self.a,))
        a_components = a_components / a_components.sum()
        return a_components

    def dataGivenA_sampler(self):
        rvs = np.empty(shape=(self.a, self.d))
        for i in range(self.a):
            rvs[i] = stats.multivariate_normal(mean=self.mu_data[i], cov=self.S[i]).rvs()
        return rvs

    def thetaGivenA_sampler(self):
        # same param prior model
        rvs = np.empty(shape=(self.a, self.n))
        for i in range(self.a):
            rvs[i] = stats.multivariate_normal(mean=self.mu_theta[i], cov=self.Sigma).rvs()
        return rvs

    def thetaGivenDataGivenA(self, D: np.ndarray, theta, a):
        mean = self.mu_theta + (torch.transpose(self.X, 1, 2) @ torch.inverse(self.S) @ (
                torch.as_tensor(D).float() - self.mu_data).unsqueeze(2)).squeeze(2)
        sigma = self.Sigma - torch.transpose(self.X, 1, 2) @ torch.inverse(self.S) @ self.X
        loglikes = np.empty(shape=(self.a,))
        for i in range(self.a):
            loglikes[i] = stats.multivariate_normal(mean=mean[i], cov=sigma[i]).logpdf(theta)
        return (np.log(a) + loglikes).sum().numpy()

    def dataGivenThetaGivenA(self, theta: np.ndarray, a: np.ndarray):
        mean = self.mu_data + (self.X @ torch.inverse(self.Sigma) @ (
                torch.as_tensor(theta).float() - self.mu_theta).unsqueeze(2)).squeeze(2)
        sigma = self.S - self.X @ self.Sigma @ torch.transpose(self.X, 1, 2)
        rv = np.empty(shape=(self.a, self.d))
        for i in range(self.a):
            rv[i] = stats.multivariate_normal(mean=mean[i], cov=sigma[i]).rvs()
        return a @ rv

    def logratio(self, x, z):
        loglike = stats.multivariate_normal(mean=(self.m + self.M @ z), cov=self.C).logpdf(x)
        logevidence = stats.multivariate_normal(mean=(self.m + self.M @ self.mu),
                                                cov=(self.C + self.M @ self.Sigma @ self.M.T)).logpdf(x)
        logratio = (loglike - logevidence).sum()
        return logratio

    def build(self, graph):
        a_i = graph.node("a_i", self.a_sampler)
        x_i = graph.node("x_i", self.dataGivenA_sampler)
        z_i = graph.node("z_i", self.thetaGivenA_sampler)
        z = graph.node(self.nreSettings.targetKey, lambda z_i, a_i: a_i @ z_i, z_i, a_i)
        x = graph.node(self.nreSettings.obsKey, self.dataGivenThetaGivenA, z_i, a_i)
        # true answer
        zgivenX_i = graph.node("l", self.thetaGivenDataGivenA, x_i, z, a_i)
