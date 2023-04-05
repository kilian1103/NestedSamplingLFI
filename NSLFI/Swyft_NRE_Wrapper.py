from typing import Dict, Any

import swyft
import torch

from NSLFI.NRE_Settings import NRE_Settings


class NRE:
    def __init__(self, network: swyft.SwyftModule, prior: Dict[str, Any],
                 nreSettings: NRE_Settings, obs: swyft.Sample, livepoints: torch.tensor):
        self.network = network.eval()
        self.livepoints = livepoints
        self.prior = prior
        self.nre_settings = nreSettings
        self.obs = {"x": torch.tensor(obs["x"]).type(torch.float64).unsqueeze(0)}

    def logLikelihood(self, proposal_sample: torch.tensor):
        # check if list of datapoints or single datapoint
        if proposal_sample.ndim == 1:
            prediction = self.network(self.obs, {"z": proposal_sample.type(torch.float64)})
            return prediction.logratios
        else:
            prediction = self.network(self.obs, {"z": proposal_sample.type(torch.float64)})
            return prediction.logratios[:, 0]
