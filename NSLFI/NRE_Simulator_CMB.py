import numpy as np
import swyft
from cmblike.cmb import CMB

from NSLFI.NRE_Settings import NRE_Settings


class Simulator(swyft.Simulator):
    def __init__(self, nreSettings: NRE_Settings, cmbs: CMB, bins, bin_centers, p_noise, cp=False):
        super().__init__()
        self.nreSettings = nreSettings
        self.cmbs = cmbs
        self.bins = bins
        self.bin_centers = bin_centers
        self.p_noise = p_noise
        self.conversion = self.bin_centers * (self.bin_centers + 1) / (2 * np.pi)
        self.cp = cp

    def prior(self):
        cube = np.random.uniform(0, 1, self.nreSettings.num_features)
        theta = self.cmbs.prior(cube=cube)
        return theta

    def likelihood(self, theta):
        cltheory, sample = self.cmbs.get_samples(self.bin_centers, theta, self.bins, noise=self.p_noise, cp=self.cp)
        return sample * self.conversion

    def build(self, graph):
        # prior
        z = graph.node(self.nreSettings.targetKey, self.prior)
        # likelihood
        x = graph.node(self.nreSettings.obsKey, self.likelihood, z)
