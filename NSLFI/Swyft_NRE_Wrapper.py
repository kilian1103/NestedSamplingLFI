import swyft
import torch

from NSLFI.NRE_Settings import NRE_Settings


class NRE:
    def __init__(self, network: swyft.SwyftModule, nreSettings: NRE_Settings, obs: swyft.Sample):
        self.network = network.eval()
        self.nre_settings = nreSettings
        self.obs = {
            self.nre_settings.obsKey: torch.tensor(obs[self.nre_settings.obsKey]).type(torch.float64).unsqueeze(0)}

    def logLikelihood(self, proposal_sample: torch.tensor):
        # check if list of datapoints or single datapoint
        if proposal_sample.ndim == 1:
            prediction = self.network(self.obs, {self.nre_settings.targetKey: proposal_sample.type(torch.float64)})
            return prediction.logratios
        else:
            prediction = self.network(self.obs, {self.nre_settings.targetKey: proposal_sample.type(torch.float64)})
            return prediction.logratios[:, 0]
