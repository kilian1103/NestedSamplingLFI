import numpy as np
import swyft
from cmblike.cmb import CMB

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings


class Simulator(swyft.Simulator):
    def __init__(self, polyswyftSettings: PolySwyft_Settings, cmbs: CMB, bins, bin_centers, p_noise, cp=False):
        super().__init__()
        self.polyswyftSettings = polyswyftSettings
        self.cmbs = cmbs
        self.bins = bins
        self.bin_centers = bin_centers
        self.p_noise = p_noise
        self.conversion = self.bin_centers * (self.bin_centers + 1) / (2 * np.pi)
        self.cp = cp

    def prior(self):
        cube = np.random.uniform(0, 1, self.polyswyftSettings.num_features)
        theta = self.cmbs.prior(cube=cube)
        return theta

    def likelihood(self, theta):
        cltheory, sample = self.cmbs.get_samples(self.bin_centers, theta, self.bins, noise=self.p_noise, cp=self.cp)
        return sample * self.conversion

    def build(self, graph):
        # prior
        z = graph.node(self.polyswyftSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.polyswyftSettings.obsKey, self.likelihood, z)
