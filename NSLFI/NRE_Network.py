import swyft
import torch

from NSLFI.NRE_Settings import NRE_Settings


class Network(swyft.SwyftModule):
    def __init__(self, nreSettings: NRE_Settings):
        super().__init__()
        self.nreSettings = nreSettings
        #  self.logratios1 = swyft.LogRatioEstimator_1dim(num_features=2, num_params=2, varnames=targetkey,
        #  dropout=0.2, hidden_features=128)
        self.optimizer_init = swyft.OptimizerInit(torch.optim.Adam, dict(lr=self.nreSettings.learning_rate_init),
                                                  torch.optim.lr_scheduler.ExponentialLR,
                                                  dict(gamma=self.nreSettings.learning_rate_decay, verbose=True))
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features=self.nreSettings.num_features_dataset, marginals=(
            tuple(dim for dim in range(self.nreSettings.num_features)),),
                                                       varnames=self.nreSettings.targetKey,
                                                       dropout=self.nreSettings.dropout, hidden_features=128, Lmax=8)

    def forward(self, A, B):
        return self.logratios2(A[self.nreSettings.obsKey], B[self.nreSettings.targetKey])
