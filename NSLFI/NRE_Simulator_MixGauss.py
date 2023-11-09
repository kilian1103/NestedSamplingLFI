import numpy as np
import scipy.stats as stats
import swyft
import torch
from lsbi.model import LinearMixtureModel

from NSLFI.NRE_Settings import NRE_Settings


class Simulator(swyft.Simulator):
    def __init__(self, nreSettings: NRE_Settings):
        super().__init__()
        self.nreSettings = nreSettings
        self.n = self.nreSettings.num_features
        self.d = self.nreSettings.num_features_dataset
        self.a = self.nreSettings.num_mixture_components
        self.a_vec = self.a_sampler()
        self.mu_theta = torch.randn(size=(self.a, self.n)) * 3  # random mean vec of parameter
        self.M = torch.randn(size=(self.a, self.d, self.n))  # random transform matrix of param to data space vec
        self.mu_data = torch.randn(size=(self.a, self.d)) * 3  # random mean vec of data
        self.Sigma = torch.eye(self.n)  # cov matrix of parameter prior
        self.S = torch.eye(self.d)  # cov matrix of dataset
        self.X = self.M @ self.Sigma  # covariance entries between data and parameter
        self.model = LinearMixtureModel(M=self.M, C=self.S, Sigma=self.Sigma, mu=self.mu_theta,
                                        m=self.mu_data, logA=np.log(self.a_vec), n=self.n, d=self.d, k=self.a)

    def a_sampler(self):
        a_components = np.random.rand(self.a)
        a_components = a_components / a_components.sum()
        return a_components

    def evidence_given_a(self, idx: int):
        mean = self.mu_data[idx] + (self.M[idx] @ self.mu_theta[idx].T).squeeze()
        cov = self.S + self.M[idx].squeeze() @ self.Sigma @ self.M[idx].squeeze().T
        return stats.multivariate_normal(mean=mean.squeeze(), cov=cov.squeeze()).rvs()

    def prior_given_a(self, idx: int):
        return stats.multivariate_normal(mean=self.mu_theta[idx].squeeze(), cov=self.Sigma.squeeze()).rvs()

    def posterior_given_a(self, D: np.ndarray, idx: int):
        cov = torch.linalg.inv(
            torch.linalg.inv(self.Sigma) + self.M[idx].T.squeeze() @ torch.linalg.inv(self.S) @ self.M[idx].squeeze())
        mean = self.mu_theta[idx].T + cov @ self.M[idx].T.squeeze() @ torch.linalg.inv(self.S) @ (
                torch.as_tensor(D).float() - self.mu_data[idx] - (self.M[idx].squeeze() @ self.mu_theta[idx].T).T).T
        rv = stats.multivariate_normal(mean=mean.squeeze(), cov=cov.squeeze()).rvs()
        return rv

    def likelihood_given_a(self, theta: np.ndarray, idx: int):
        mean = self.mu_data[idx] + self.M[idx].squeeze() @ torch.as_tensor(theta).float()
        cov = self.S
        rv = stats.multivariate_normal(mean=mean.squeeze(), cov=cov.squeeze()).rvs()
        return rv

    def logratio(self, x, post):
        z, w = post
        logratio = self.model.likelihood(z).logpdf(x) - self.model.evidence().logpdf(x)
        return logratio

    def posterior(self, x):
        post = self.model.posterior(x)
        z = post.rvs()
        posterior = (z, post.logpdf(z))
        return posterior

    def prior(self):
        return self.model.prior().rvs()

    def likelihood(self, z):
        return self.model.likelihood(z).rvs()

    def build(self, graph):
        # i = graph.node("i", np.random.choice, self.a, 1, True, self.a_vec)
        # prior
        # z_i = graph.node("z_i", self.prior_given_a, i)
        z = graph.node(self.nreSettings.targetKey, self.prior)
        # likelihood
        # x_i = graph.node("x_i", self.likelihood_given_a, z_i, i)
        x = graph.node(self.nreSettings.obsKey, self.likelihood, z)
        # posterior
        # post_i = graph.node("post_i", self.posterior_given_a, x_i, i)
        post = graph.node(self.nreSettings.posteriorsKey, self.posterior, x)
        # logratio
        l = graph.node(self.nreSettings.contourKey, self.logratio, x, post)
