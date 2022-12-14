import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import swyft
from anesthetic import NestedSamples
from scipy.stats import multivariate_normal

import NSLFI.NestedSampler
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.Swyft_NRE_Wrapper import NRE


def execute():
    np.random.seed(234)
    logging.basicConfig(filename="myLFI.log", level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    root = "swyft_data"  # filepath
    # try creating new root folder for data storage
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    nreSettings = NRE_Settings(base_path=root)
    nreSettings.n_training_samples = 10_00
    nreSettings.n_weighted_samples = 10_000
    nreSettings.trainmode = True
    # NRE model path
    checkpoint = "lightning_logs/version_7/checkpoints/epoch=19-step=80.ckpt"
    # define forward model dimensions
    nParam = 2
    nData = 3

    # NS rounds 0 is default
    rounds = 0

    class Simulator(swyft.Simulator):
        def __init__(self):
            super().__init__()
            self.transform_samples = swyft.to_numpy32

        def build(self, graph):
            means = graph.node('means', lambda: theta_prior.rvs(size=nParam))
            x = graph.node('x',
                           lambda means: multivariate_normal.rvs(F @ means, cov),
                           means)

    sim = Simulator()
    samples = sim.sample(N=nreSettings.n_training_samples)

    # initialize swyft network
    class Network(swyft.SwyftModule):
        def __init__(self):
            super().__init__()
            marginals = (tuple(x for x in range(nParam)),)
            # self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 1, num_params = 3, varnames = 'z')
            self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features=nData, marginals=marginals,
                                                           varnames='means',
                                                           hidden_features=32)

        def forward(self, A, B):
            # logratios1 = self.logratios1(A['x'], B['z'])
            logratios2 = self.logratios2(A['x'], B['means'])
            return logratios2

    trainer = swyft.SwyftTrainer(accelerator=nreSettings.device, devices=1, max_epochs=20, precision=64,
                                 enable_progress_bar=False, default_root_dir=nreSettings.base_path)
    dm = swyft.SwyftDataModule(samples, fractions=[0.8, 0.02, 0.1], num_workers=3, batch_size=256)
    network = Network()

    # train NRE
    if nreSettings.trainmode is True:
        trainer.fit(network, dm)
    # load NRE from file
    else:
        checkpoint_path = os.path.join(nreSettings.base_path,
                                       checkpoint)
        network = network.load_from_checkpoint(checkpoint_path)

    # get posterior samples
    means = theta_prior.rvs(size=(nreSettings.n_weighted_samples, nParam))
    B = swyft.Samples(means=means)
    C = swyft.Sample(means=x0)
    x0 = swyft.Sample(x=x0)
    predictions = trainer.infer(network, x0, B)
    # plot initial NRE

    plt.figure()
    labeler = {f"means[{i}]": fr"$\mu_{i}$" for i in range(nParam)}
    swyft.corner(predictions, tuple(f"means[{i}]" for i in range(nParam)), labeler=labeler, bins=200, smooth=3);
    plt.suptitle("NRE parameter estimation")
    plt.savefig(fname=f"{nreSettings.base_path}/firstNRE.pdf")
    logProb_0 = trainer.infer(network, x0, C)
    logger.info(f"log probability of theta_0 using NRE is: {float(logProb_0.logratios):.3f}")

    # wrap NRE object
    trained_NRE = NRE(network=network, trainer=trainer, prior=prior, nreSettings=nreSettings, obs=x0,
                      livepoints=means)

    # # wrap NRE for Polychord
    # poly_NRE = NRE_Poly(nre=trained_NRE.mre_2d, x0=x0)
    # polychordSet = PolyChordSettings(nDims=poly_NRE.nDims, nDerived=poly_NRE.nDerived)
    # polychordSet.nlive = n_training_samples
    # try:
    #     comm_analyse = MPI.COMM_WORLD
    #     rank_analyse = comm_analyse.Get_rank()
    # except Exception as e:
    #     logger.error(
    #         "Oops! {} occurred. when Get_rank()".format(e.__class__))
    #     rank_analyse = 0
    #
    #
    # def dumper(live, dead, logweights, logZ, logZerr):
    #     """Dumper Function for PolyChord for runtime progress access."""
    #     logger.info("Last dead point: {}".format(dead[-1]))
    #
    #
    # output = pypolychord.run_polychord(poly_NRE.loglike,
    #                                    poly_NRE.nDims,
    #                                    poly_NRE.nDerived,
    #                                    polychordSet,
    #                                    poly_NRE.prior, dumper)
    # comm_analyse.Barrier()

    # optimize with my NS run
    output = NSLFI.NestedSampler.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                                 livepoints=trained_NRE.livepoints, prior=prior, nsim=100,
                                                 stop_criterion=1e-3,
                                                 samplertype="Metropolis", root=nreSettings.base_path, rounds=rounds)
    logger.info(output)
    if rounds == 0:
        deadpoints = np.load(file=f"{nreSettings.base_path}/posterior_samples.npy")
        weights = np.load(file=f"{nreSettings.base_path}/weights.npy")
        deadpoints_birthlogL = np.load(file=f"{nreSettings.base_path}/logL_birth.npy")
        deadpoints_logL = np.load(file=f"{nreSettings.base_path}/logL.npy")
        nested = NestedSamples(data=deadpoints, weights=weights, logL_birth=deadpoints_birthlogL,
                               logL=deadpoints_logL, root=nreSettings.base_path)
        plt.figure()
        nested.plot_2d([0, 1])
        plt.suptitle("NRE NS enhanced samples")
        plt.savefig(fname=f"{nreSettings.base_path}/afterNS.pdf")
    else:
        firstRoundPoints = means
        secondRoundPoints = np.load(file=f"{root}/posterior_samples_rounds_0.npy")
        thirdRoundPoints = np.load(file=f"{root}/posterior_samples_rounds_1.npy")

        plt.figure()
        plt.scatter(firstRoundPoints[:, 0], firstRoundPoints[:, 1], c="b", label="Round 1", s=3)
        plt.scatter(secondRoundPoints[:, 0], secondRoundPoints[:, 1], c="r", label="Round 2", s=3)
        plt.scatter(thirdRoundPoints[:, 0], thirdRoundPoints[:, 1], c="y", label="Round 3", s=3)
        plt.legend()
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.vlines(x=0.5, ymin=loc, ymax=scale, colors="cyan")
        plt.hlines(y=0.5, xmin=loc, xmax=scale, colors="cyan")
        plt.title(r"Log-weights contours using NS rounds, $\mathcal{L} > "
                  r"\mathcal{L}_{\mathrm{median}}$")
        plt.savefig(f"{root}/NS_rounds.pdf")

    logger.info("Done")


if __name__ == '__main__':
    execute()
