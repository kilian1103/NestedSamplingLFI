class NRE:
    def __init__(self, dataset, store, prior, priorLimits, trainedNRE, nreSettings, posterior):
        self.dataset = dataset
        self.store = store
        self.prior = prior
        self.priorLimits = priorLimits
        self.nre_settings = nreSettings
        self.mre_3d = trainedNRE
        self.marginal_indices_3d = (0, 1, 2)
        self.posterior = posterior
