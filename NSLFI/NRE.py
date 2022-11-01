from typing import Dict

import swyft
import numpy as np
from NSLFI.NRE_Settings import NRE_Settings


class NRE:
    def __init__(self, dataset: swyft.Dataset, store: swyft.Store, prior: swyft.Prior, priorLimits: Dict[str, float],
                 trainedNRE: swyft.MarginalRatioEstimator, nreSettings: NRE_Settings, x_0: np.ndarray):
        self.dataset = dataset
        self.store = store
        self.prior = prior
        self.priorLimits = priorLimits
        self.nre_settings = nreSettings
        self.mre_2d = trainedNRE
        self.x_0 = x_0
        self.marginal_indices_2d = (0, 1)


    def logLikelihood(self, proposal_sample, ndim):
        return self.mre_2d.log_ratio(observation=self.x_0, v=[proposal_sample])[
            self.marginal_indices_2d].copy()
