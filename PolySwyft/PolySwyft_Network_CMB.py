from typing import Tuple, List, Any

import numpy as np
import swyft
import torch
from cmblike.cmb import CMB

from PolySwyft.PolySwyft_Settings import NRE_Settings


class Network(swyft.SwyftModule):
    def __init__(self, nreSettings: NRE_Settings, obs: swyft.Sample, cmbs: CMB):
        super().__init__()
        self.nreSettings = nreSettings
        self.obs = obs
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr=self.nreSettings.learning_rate_init),
                                                  torch.optim.lr_scheduler.ExponentialLR,
                                                  dict(gamma=self.nreSettings.learning_rate_decay))
        self.network = swyft.LogRatioEstimator_Ndim(num_features=self.nreSettings.num_summary_features, marginals=(
            tuple(dim for dim in range(self.nreSettings.num_features)),),
                                                    varnames=self.nreSettings.targetKey,
                                                    dropout=self.nreSettings.dropout, hidden_features=64, Lmax=0,
                                                    num_blocks=3)
        self.cmbs = cmbs
        self.summarizer = torch.nn.Sequential(torch.nn.Linear(self.nreSettings.num_features_dataset, 32),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(32, 32),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(32, 16),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(16, self.nreSettings.num_summary_features)
                                              )

    def forward(self, A, B):
        s = self.summarizer(A[self.nreSettings.obsKey])
        return self.network(s, B[self.nreSettings.targetKey])

    def prior(self, cube) -> np.ndarray:
        """Transforms the unit cube to the prior cube."""
        theta = self.cmbs.prior(cube=cube)
        return theta

    def logLikelihood(self, theta: np.ndarray) -> Tuple[Any, List]:
        """Computes the loglikelihood ("NRE") of the given theta."""
        theta = torch.as_tensor(theta)
        # check if list of datapoints or single datapoint
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        s = self.summarizer(self.obs[self.nreSettings.obsKey])
        prediction = self.network(s, theta)
        if prediction.logratios[:, 0].shape[0] == 1:
            return float(prediction.logratios[:, 0]), []
        else:
            return prediction.logratios[:, 0], []

    def dumper(self, live, dead, logweights, logZ, logZerr):
        """Dumper Function for PolyChord for runtime progress access."""
        print("Last dead point: {}".format(dead[-1]))

    def get_new_network(self):
        return Network(nreSettings=self.nreSettings, obs=self.obs, cmbs=self.cmbs)
