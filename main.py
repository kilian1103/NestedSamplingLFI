import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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
    nreSettings = NRE_Settings(base_path="swyft_data/")
    nreSettings.n_training_samples = 10_00
    nreSettings.n_weighted_samples = 10_000

    ndim = 2
    prior = {f"theta_{i}": stats.uniform(loc=0, scale=1) for i in range(ndim)}

    ### swyft mode###
    mode = "train"
    device = nreSettings.device

    # true parameters of simulator
    x0 = np.array([0.5 for x in range(ndim)])

    # saving file names
    observation_filename = nreSettings.observation_filename

    # define forward model
    # ndim parameter space we try to infer
    nData = 1
    cov = 0.05 * np.eye(ndim)

    class Simulator(swyft.Simulator):
        def __init__(self):
            super().__init__()
            self.transform_samples = swyft.to_numpy32

        def build(self, graph):
            means = graph.node('means', lambda: np.random.rand(ndim))
            x = graph.node('x', lambda means: multivariate_normal.rvs(mean=means, cov=cov, size=nData), means)

    sim = Simulator()
    samples = sim.sample(N=nreSettings.n_training_samples)

    # plot observation

    # initialize swyft

    class Network(swyft.SwyftModule):
        def __init__(self):
            super().__init__()
            marginals = (tuple(x for x in range(ndim)),)
            # self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 1, num_params = 3, varnames = 'z')
            self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features=ndim, marginals=marginals, varnames='means')

        def forward(self, A, B):
            # logratios1 = self.logratios1(A['x'], B['z'])
            logratios2 = self.logratios2(A['x'], B['means'])
            return logratios2

    trainer = swyft.SwyftTrainer(accelerator=device, devices=1, max_epochs=3, precision=64, logger=False,
                                 enable_progress_bar=False)
    dm = swyft.SwyftDataModule(samples, fractions=[0.8, 0.02, 0.1], num_workers=3, batch_size=256)
    network = Network()

    # train NRE
    trainer.fit(network, dm)
    # load MRE from file

    # get posterior samples
    means = np.random.rand(nreSettings.n_weighted_samples, ndim)
    B = swyft.Samples(means=means)
    theta_0 = swyft.Sample(x=x0)
    C = swyft.Sample(means=x0)
    predictions = trainer.infer(network, theta_0, B)
    # plot initial NRE

    plt.figure()
    labeler = {f"means[{i}]": fr"$\mu_{i}$" for i in range(ndim)}
    swyft.corner(predictions, tuple(f"means[{i}]" for i in range(ndim)), labeler=labeler, bins=200, smooth=3);
    plt.suptitle("NRE parameter estimation")
    plt.savefig(fname="swyft_data/firstNRE.pdf")
    logProb_0 = trainer.infer(network, theta_0, C)
    logger.info(f"log probability of theta_0 using NRE is: {float(logProb_0.logratios):.3f}")

    # wrap NRE object
    trained_NRE = NRE(network=network, trainer=trainer, prior=prior, nreSettings=nreSettings, obs=theta_0,
                      livepoints=means)

    # # wrap NRE for Polychord
    # poly_NRE = NRE_Poly(nre=trained_NRE.mre_2d, x_0=x_0)
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
                                                 samplertype="Metropolis")
    logger.info(output)

    deadpoints = np.load(file="posterior_samples.npy")
    weights = np.load(file="weights.npy")
    deadpoints_birthlogL = np.load(file="logL_birth.npy")
    deadpoints_logL = np.load(file="logL.npy")
    nested = NestedSamples(data=deadpoints, weights=weights, logL_birth=deadpoints_birthlogL,
                           logL=deadpoints_logL)
    plt.figure()
    nested.plot_2d([0, 1])
    plt.suptitle("NRE NS enhanced samples")
    plt.savefig(fname="swyft_data/afterNS.pdf")

    # add datapoint to NRE

    # retrain NRE

    logger.info("Done")


if __name__ == '__main__':
    execute()
