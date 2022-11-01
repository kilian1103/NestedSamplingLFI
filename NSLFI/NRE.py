from typing import Dict

import numpy as np
import swyft

from NSLFI.NRE_Settings import NRE_Settings


class NRE:
    def __init__(self, dataset: swyft.Dataset, store: swyft.Store, prior: swyft.Prior, priorLimits: Dict[str, float],
                 trained_NRE: swyft.MarginalRatioEstimator, nreSettings: NRE_Settings, x_0: np.ndarray):
        self.dataset = dataset
        self.store = store
        self.prior = prior
        self.priorLimits = priorLimits
        self.nre_settings = nreSettings
        self.mre_2d = trained_NRE
        self.x_0 = x_0
        self.marginal_indices_2d = (0, 1)
        self.ndim = len(self.marginal_indices_2d)

    def logLikelihood(self, proposal_sample: np.ndarray, ndim: int):
        # check if list of datapoints or single datapoint
        if proposal_sample.ndim == 1:
            proposal_sample = [proposal_sample]
        return self.mre_2d.log_ratio(observation=self.x_0, v=proposal_sample)[
            self.marginal_indices_2d].copy()
