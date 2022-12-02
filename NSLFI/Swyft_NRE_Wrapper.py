from typing import Dict, Any

import numpy as np
import swyft

from NSLFI.NRE_Settings import NRE_Settings


class NRE:
    def __init__(self, network: swyft.SwyftModule, trainer: swyft.SwyftTrainer, prior: Dict[str, Any],
                 nreSettings: NRE_Settings, obs: swyft.Sample, livepoints: np.ndarray):
        self.trainer = trainer
        self.network = network
        self.livepoints = livepoints
        self.prior = prior
        self.nre_settings = nreSettings
        self.obs = obs

    def logLikelihood(self, proposal_sample: np.ndarray):
        # check if list of datapoints or single datapoint
        if proposal_sample.ndim == 1:
            proposal_sample = swyft.Sample(means=proposal_sample)
            prediction = self.trainer.infer(self.network, self.obs, proposal_sample)
            return float(prediction.logratios)
        else:
            proposal_sample = swyft.Samples(means=proposal_sample)
            prediction = self.trainer.infer(self.network, self.obs, proposal_sample)
            return prediction.logratios[:, 0].numpy()
