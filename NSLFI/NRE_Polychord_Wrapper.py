from typing import Tuple, List, Any

import numpy as np
import swyft
import torch
from pypolychord.priors import UniformPrior

from NSLFI.NRE_Network import Network


class NRE_PolyChord:
    """Wrapper for the NRE to be used with PolyChord."""

    def __init__(self, network: Network, obs: swyft.Sample):
        """Initializes the NRE_PolyChord."""
        self.network = network.eval()
        self.nre_settings = self.network.nreSettings
        self.obs = {
            self.nre_settings.obsKey: torch.tensor(obs[self.nre_settings.obsKey]).type(torch.float64).unsqueeze(0)}

    def prior(self, cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        theta = np.zeros_like(cube)
        theta[0] = UniformPrior(self.nre_settings.sim_prior_lower,
                                self.nre_settings.sim_prior_lower + self.nre_settings.prior_width)(
            cube[0])
        theta[1] = UniformPrior(self.nre_settings.sim_prior_lower,
                                self.nre_settings.sim_prior_lower + self.nre_settings.prior_width)(
            cube[1])
        return theta

    def logLikelihood(self, theta: np.ndarray) -> Tuple[Any, List]:
        """Computes the loglikelihood ("NRE") of the given theta."""
        theta = torch.as_tensor(theta)
        # check if list of datapoints or single datapoint
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        prediction = self.network(self.obs, {self.nre_settings.targetKey: theta.type(torch.float64)})
        if prediction.logratios[:, 0].shape[0] == 1:
            return float(prediction.logratios[:, 0]), []
        else:
            return prediction.logratios[:, 0], []

    def dumper(self, live, dead, logweights, logZ, logZerr):
        """Dumper Function for PolyChord for runtime progress access."""
        print("Last dead point: {}".format(dead[-1]))
