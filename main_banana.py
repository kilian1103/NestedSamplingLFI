import logging
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import swyft
from swyft.lightning.utils import collate_output

import NSLFI.NestedSampler
from NSLFI.NRE_Settings import NRE_Settings


# from NSLFI.Swyft_NRE_Wrapper import NRE
def execute():
    np.random.seed(234)
    logging.basicConfig(filename="myLFI.log", level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    root = "swyft_banana_unimodal_Slice_30k_NRE_retrain"
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")

    nreSettings = NRE_Settings(base_path=root)
    nreSettings.n_training_samples = 30_000
    nreSettings.n_weighted_samples = 10_000
    nreSettings.trainmode = True
    # NS rounds, 0 is default NS run
    rounds = 1
    # Retrain rounds
    retrain_rounds = 2
    samplerType = "Slice"
    # define forward model dimensions
    bimodal = False
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
                                                           dropout=0.2, hidden_features=128, Lmax=8)

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
            f"Christoph/banana_problem_unimodal_NRE_nd_model/lightning_logs/version_0/checkpoints/epoch=9-step"
            f"=3750.ckpt")
        network = network.load_from_checkpoint(checkpoint_path)
    # get posterior samples
    prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=['z'])
    predictions = trainer.infer(network, obs, prior_samples)
    plt.figure()
    swyft.corner(predictions, ["z[0]", "z[1]"], bins=50, smooth=1)
    plt.savefig(f"{root}/NRE_predictions.pdf")
    plt.show()

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
                                                 samplertype=samplerType, rounds=rounds, root=root,
                                                 iter=nreSettings.n_training_samples)

    prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=['z'])
    predictions = trainer.infer(network, obs, prior_samples)

    def retrain_next_round(root: str, nextRoundPoints: np.ndarray):
        try:
            os.makedirs(root)
        except OSError:
            logger.info("root folder already exists!")
        out = []
        for z in nextRoundPoints:
            trace = dict()
            trace["z"] = z
            sim.graph["x"].evaluate(trace)
            sim.graph["l"].evaluate(trace)
            result = sim.transform_samples(trace)
            out.append(result)
        out = collate_output(out)
        nextRoundSamples = swyft.Samples(out)

        trainer = swyft.SwyftTrainer(accelerator='cpu', devices=1, max_epochs=10, precision=64,
                                     enable_progress_bar=False,
                                     default_root_dir=root)
        dm = swyft.SwyftDataModule(nextRoundSamples, fractions=[0.8, 0.1, 0.1], num_workers=0, batch_size=64)
        network = Network()
        trainer.fit(network, dm)
        # get posterior samples
        prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=['z'])
        predictions = trainer.infer(network, obs, prior_samples)
        plt.figure()
        swyft.corner(predictions, ["z[0]", "z[1]"], bins=50, smooth=1)
        plt.savefig(f"{root}/NRE_predictions.pdf")
        plt.show()
        # wrap NRE object
        trained_NRE = NRE(network=network, trainer=trainer, prior=prior, nreSettings=nreSettings, obs=obs,
                          livepoints=nextRoundSamples["z"])
        output = NSLFI.NestedSampler.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                                     livepoints=trained_NRE.livepoints, prior=prior, nsim=100,
                                                     stop_criterion=1e-3, rounds=1,
                                                     root=root,
                                                     samplertype=samplerType,
                                                     iter=nreSettings.n_training_samples)

    for rd in range(1, retrain_rounds + 1):
        nextRoundPoints = np.load(file=f"{root}/posterior_samples_rounds_0.npy")
        newRoot = root + f"_rd_{rd}"
        retrain_next_round(newRoot, nextRoundPoints)
        root = newRoot


if __name__ == '__main__':
    execute()
