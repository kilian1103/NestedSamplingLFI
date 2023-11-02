from typing import Tuple, List, Any

import numpy as np
import swyft
import torch
from pypolychord.priors import UniformPrior

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Settings import NRE_Settings


class NRE_PolyChord(Network):
    """Wrapper for the NRE to be used with PolyChord."""

    def __init__(self, network: Network, obs: swyft.Sample, nreSettings: NRE_Settings):
        """Initializes the NRE_PolyChord."""
        super().__init__(nreSettings=nreSettings)
        self.network = network.eval()
        self.obs = obs

    def prior(self, cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        theta = np.zeros_like(cube)
        for i in range(len(cube)):
            theta[i] = UniformPrior(-2, 2)(cube[i])
        return theta

    def logLikelihood(self, theta: np.ndarray) -> Tuple[Any, List]:
        """Computes the loglikelihood ("NRE") of the given theta."""
        theta = torch.tensor(theta)
        # check if list of datapoints or single datapoint
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        prediction = self.network(self.obs, {self.nreSettings.targetKey: theta})
        if prediction.logratios[:, 0].shape[0] == 1:
            return float(prediction.logratios[:, 0]), []
        else:
            return prediction.logratios[:, 0], []

    def dumper(self, live, dead, logweights, logZ, logZerr):
        """Dumper Function for PolyChord for runtime progress access."""
        print("Last dead point: {}".format(dead[-1]))

    def set_network(self, network: Network):
        """Sets a network for PolySwyft."""
        self.network = network.eval()
