from typing import Tuple, List, Any

import numpy as np
import swyft
import torch

from NSLFI.NRE_Settings import NRE_Settings


class Network(swyft.SwyftModule):
    def __init__(self, nreSettings: NRE_Settings, obs: swyft.Sample, **kwargs):
        super().__init__()
        self.flow = kwargs.pop("flow", None)
        self.nreSettings = nreSettings
        self.obs = obs
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr=self.nreSettings.learning_rate_init),
                                                  torch.optim.lr_scheduler.ExponentialLR,
                                                  dict(gamma=self.nreSettings.learning_rate_decay, verbose=True))
        self.network = swyft.LogRatioEstimator_Ndim(num_features=self.nreSettings.num_features_dataset, marginals=(
            tuple(dim for dim in range(self.nreSettings.num_features)),),
                                                    varnames=self.nreSettings.targetKey,
                                                    dropout=self.nreSettings.dropout, hidden_features=128, Lmax=0)

    def forward(self, A, B):
        return self.network(A[self.nreSettings.obsKey], B[self.nreSettings.targetKey])

    def prior(self, cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        return self.nreSettings.flow(cube.reshape(1, self.nreSettings.num_features)).numpy().squeeze()

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

    def set_network(self, network: swyft.SwyftModule):
        """Sets a network for PolySwyft."""
        self.network = network.eval()

    def get_new_network(self):
        return Network(nreSettings=self.nreSettings, obs=self.obs)
