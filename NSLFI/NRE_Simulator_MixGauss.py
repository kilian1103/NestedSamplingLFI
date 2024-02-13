import numpy as np
import swyft
import torch
from lsbi.model import LinearMixtureModel

from NSLFI.NRE_Settings import NRE_Settings


class Simulator(swyft.Simulator):
    def __init__(self, nreSettings: NRE_Settings, mu_theta: torch.Tensor, M: torch.Tensor, mu_data: torch.Tensor,
                 Sigma: torch.Tensor, C: torch.Tensor):
        super().__init__()
        self.nreSettings = nreSettings
        self.n = self.nreSettings.num_features
        self.d = self.nreSettings.num_features_dataset
        self.a = self.nreSettings.num_mixture_components
        self.a_vec = self.a_sampler()
        self.X = M @ Sigma
        self.model = LinearMixtureModel(M=M, C=C, Sigma=Sigma, mu=mu_theta,
                                        m=mu_data, logA=np.log(self.a_vec), n=self.n, d=self.d, k=self.a)

    def a_sampler(self):
        a_components = np.random.rand(self.a)
        a_components = a_components / a_components.sum()
        return a_components

    def logratio(self, x, z):
        logratio = self.model.likelihood(z).logpdf(x) - self.model.evidence().logpdf(x)
        return logratio

    def posterior(self, x):
        post = self.model.posterior(x)
        return post.rvs()

    def prior(self):
        return self.model.prior().rvs()

    def likelihood(self, z):
        return self.model.likelihood(z).rvs()

    def build(self, graph):
        # prior
        z = graph.node(self.nreSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.nreSettings.obsKey, self.likelihood, z)
        # posterior
        post = graph.node(self.nreSettings.posteriorsKey, self.posterior, x)
        # logratio
        l = graph.node(self.nreSettings.contourKey, self.logratio, x, post)
