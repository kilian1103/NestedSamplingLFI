import swyft
import torch
from torch import Tensor

from NSLFI.NRE_Network import Network


class NRE:
    def __init__(self, network: Network, obs: swyft.Sample, livepoints: Tensor):
        self.network = network.eval()
        self.nre_settings = self.network.nreSettings
        self.livepoints = livepoints
        self.obs = {
            self.nre_settings.obsKey: torch.tensor(obs[self.nre_settings.obsKey]).type(torch.float64).unsqueeze(0)}

    def logLikelihood(self, proposal_sample: Tensor) -> Tensor:
        # check if list of datapoints or single datapoint
        if proposal_sample.ndim == 1:
            prediction = self.network(self.obs, {self.nre_settings.targetKey: proposal_sample.type(torch.float64)})
            return prediction.logratios
        else:
            prediction = self.network(self.obs, {self.nre_settings.targetKey: proposal_sample.type(torch.float64)})
            return prediction.logratios[:, 0]