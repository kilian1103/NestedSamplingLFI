import logging
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import swyft

import NSLFI.NestedSampler
from NSLFI.NRE_Settings import NRE_Settings


# from NSLFI.Swyft_NRE_Wrapper import NRE
def execute():
    np.random.seed(234)
    logging.basicConfig(filename="myLFI.log", level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    root = "swyft_banana_bimodal_Slice"
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")

    nreSettings = NRE_Settings(base_path=root)
    nreSettings.n_training_samples = 30_000
    nreSettings.n_weighted_samples = 10_000
    nreSettings.trainmode = False
    # NS rounds, 0 is default NS run
    rounds = 2
    bimodal = True
    # define forward model dimensions
    nParam = 2
    # true parameters of simulator
    obs = swyft.Sample(x=np.array(nParam * [0]))
    # uniform prior for theta_i
    lower = -1
    upper = 2
    theta_prior = stats.uniform(loc=lower, scale=upper)
    # wrap prior for NS sampling procedure
    prior = {f"theta_{i}": theta_prior for i in range(nParam)}

    class Simulator(swyft.Simulator):
        def __init__(self, bounds_z=None, bimodal=bimodal):
            super().__init__()
            self.z_sampler = swyft.RectBoundSampler(
                [stats.uniform(lower, upper),
                 stats.uniform(lower, upper),
                 ],
                bounds=bounds_z
            )
            self.bimodal = bimodal

        def f(self, z):
            if self.bimodal:
                if z[0] < 0:
                    z = np.array([z[0] + 0.5, z[1] - 0.5])
                else:
                    z = np.array([z[0] - 0.5, -z[1] - 0.5])
            z = 10 * np.array([z[0], 10 * z[1] + 100 * z[0] ** 2])
            return z

        def build(self, graph):
            z = graph.node("z", self.z_sampler)
            x = graph.node("x", lambda z: self.f(z) + np.random.randn(2), z)
            l = graph.node("l", lambda z: -stats.norm.logpdf(self.f(z)).sum(),
                           z)  # return -ln p(x=0|z) for cross-checks

    sim = Simulator(bounds_z=None, bimodal=bimodal)
    samples = sim.sample(nreSettings.n_training_samples)
    dm = swyft.SwyftDataModule(samples, fractions=[0.8, 0.1, 0.1], num_workers=0, batch_size=64)
    plt.tricontour(samples['z'][:, 0], samples['z'][:, 1], samples['l'] - samples['l'].min(), levels=[0, 1, 4])
    if bimodal:
        plt.ylim(-0.55, 0.55)
    else:
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.8, 0.1)

    class Network(swyft.SwyftModule):
        def __init__(self):
            super().__init__()
            #  self.logratios1 = swyft.LogRatioEstimator_1dim(num_features=2, num_params=2, varnames='z',
            #  dropout=0.2, hidden_features=128)
            self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features=2, marginals=((0, 1),), varnames='z',
                                                           dropout=0.2, hidden_features=128)

        def forward(self, A, B):
            return self.logratios2(A['x'], B['z'])

    network = Network()
    trainer = swyft.SwyftTrainer(accelerator='cpu', devices=1, max_epochs=10, precision=64, enable_progress_bar=False,
                                 default_root_dir=nreSettings.base_path)
    # train MRE
    if nreSettings.trainmode is True:
        trainer.fit(network, dm)

    # load NRE from file
    else:
        checkpoint_path = os.path.join(
            f"Christoph/banana_problem_bimodal_NRE_nd_model/lightning_logs/version_0/checkpoints/epoch=9-step"
            f"=3750.ckpt")
        network = network.load_from_checkpoint(checkpoint_path)
    # get posterior samples
    prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=['z'])
    predictions = trainer.infer(network, obs, prior_samples)
    swyft.corner(predictions, ["z[0]", "z[1]"], bins=50, smooth=1)

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
                proposal_sample = swyft.Sample(z=proposal_sample)
                prediction = self.trainer.infer(self.network, self.obs, proposal_sample)
                return float(prediction.logratios)
            else:
                proposal_sample = swyft.Samples(z=proposal_sample)
                prediction = self.trainer.infer(self.network, self.obs, proposal_sample)
                return prediction.logratios[:, 0].numpy()

    # wrap NRE object
    trained_NRE = NRE(network=network, trainer=trainer, prior=prior, nreSettings=nreSettings, obs=obs,
                      livepoints=samples["z"])

    output = NSLFI.NestedSampler.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                                 livepoints=trained_NRE.livepoints, prior=prior, nsim=100,
                                                 stop_criterion=1e-3,
                                                 samplertype="Slice", rounds=rounds, root=root)
    firstRoundPoints = samples["z"]
    secondRoundPoints = np.load(file=f"{root}/posterior_samples_rounds_0.npy")
    secondRoundPoints_birthLog = np.load(file=f"{root}/logL_birth_rounds_0.npy")[0]
    secondRoundPoints_LogLike = np.load(file=f"{root}/logL_rounds_0.npy")
    thirdRoundPoints = np.load(file=f"{root}/posterior_samples_rounds_1.npy")
    thirdRoundPoints_birthLog = np.load(file=f"{root}/logL_birth_rounds_1.npy")[0]
    thirdRoundPoints_LogLike = np.load(file=f"{root}/logL_rounds_1.npy")

    np.median(secondRoundPoints_LogLike)
    thirdRoundPoints_LogLike.min()
    plt.figure()
    plt.scatter(firstRoundPoints[:, 0], firstRoundPoints[:, 1], c="b", label="Round 0", s=3)
    plt.scatter(secondRoundPoints[:, 0], secondRoundPoints[:, 1], c="r", label="Round 1", s=3)
    plt.scatter(thirdRoundPoints[:, 0], thirdRoundPoints[:, 1], c="y", label="Round 2", s=3)
    plt.legend()
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    # plt.vlines(x=0, ymin=-1, ymax=1, colors="cyan")
    # plt.hlines(y=0, xmin=-1, xmax=1, colors="cyan")
    plt.title(r"Log-weights contours using NS rounds, $\mathcal{L} > "
              r"\mathcal{L}_{\mathrm{median}}$")
    plt.savefig(f"{root}/NS_rounds.pdf")


if __name__ == '__main__':
    execute()
