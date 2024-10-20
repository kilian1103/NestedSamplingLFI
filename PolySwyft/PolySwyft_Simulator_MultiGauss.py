import swyft
import torch
from lsbi.model import LinearModel

from PolySwyft.PolySwyft_Settings import NRE_Settings


class Simulator(swyft.Simulator):
    def __init__(self, nreSettings: NRE_Settings, m: torch.Tensor, M: torch.Tensor, C: torch.Tensor, mu: torch.Tensor,
                 Sigma: torch.Tensor):
        super().__init__()
        self.nreSettings = nreSettings
        self.n = self.nreSettings.num_features
        self.d = self.nreSettings.num_features_dataset
        self.C_inv = torch.inverse(C)
        self.model = LinearModel(M=M, C=C, Sigma=Sigma, mu=mu, m=m, n=self.n, d=self.d)

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
