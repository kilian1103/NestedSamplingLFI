import swyft


class Swyft_NRE_Poly:
    def __init__(self, trainer: swyft.SwyftTrainer, network: swyft.SwyftModule, obs: swyft.Sample):
        self.trainer = trainer
        self.network = network
        self.obs = obs

    def loglike(self, theta):
        proposal_sample = swyft.Sample(means=theta)
        prediction = self.trainer.infer(self.network, self.obs, proposal_sample)
        loglikelihood = float(prediction.logratios)
        r2 = 0
        return loglikelihood, [r2]
