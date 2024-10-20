import numpy as np
import scipy.stats as stats
import swyft

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


class Simulator(swyft.Simulator):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, bounds_z=None, bimodal=True):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.z_sampler = swyft.RectBoundSampler(
            [stats.uniform(self.polyswyftSettings.sim_prior_lower, self.polyswyftSettings.prior_width),
             stats.uniform(self.polyswyftSettings.sim_prior_lower, self.polyswyftSettings.prior_width),
             ],
            bounds=bounds_z
        )
        self.bimodal = bimodal

    def f(self, z):
        if self.bimodal:
            if z[0] < 0:
                z = np.array([z[0] + 0.5, z[1] - 0.5])
            else:
                z = np.array([z[0] - 0.5, -z[1] - 0.5])
        z = 10 * np.array([z[0], 10 * z[1] + 100 * z[0] ** 2])
        return z

    def build(self, graph):
        z = graph.node(self.polyswyftSettings.targetKey, self.z_sampler)
        x = graph.node(self.polyswyftSettings.obsKey, lambda z: self.f(z) + np.random.randn(self.polyswyftSettings.num_features), z)
        l = graph.node(self.polyswyftSettings.contourKey, lambda z: -stats.norm.logpdf(self.f(z)).sum(),
                       z)  # return -ln p(x=0|z) for cross-checks
