from typing import Dict

import swyft

from NSLFI.NRE_Settings import NRE_Settings


class NRE:
    def __init__(self, dataset: swyft.Dataset, store: swyft.Store, prior: swyft.Prior, priorLimits: Dict[str, float],
                 trainedNRE: swyft.MarginalRatioEstimator, nreSettings: NRE_Settings):
        self.dataset = dataset
        self.store = store
        self.prior = prior
        self.priorLimits = priorLimits
        self.nre_settings = nreSettings
        self.mre_3d = trainedNRE
        self.marginal_indices_3d = (0, 1, 2)
