import logging

import pypolychord
import swyft
import torch
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Post_Analysis import plot_analysis_of_NSNRE
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator_MixGauss import Simulator
from NSLFI.PolySwyft import PolySwyft
from NSLFI.utils import reload_data_for_plotting


def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    nreSettings = NRE_Settings()
    seed_everything(nreSettings.seed, workers=True)
    logging.basicConfig(filename=nreSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    nreSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    n = nreSettings.num_features
    d = nreSettings.num_features_dataset
    a = nreSettings.num_mixture_components

    mu_data = torch.randn(size=(a, d)) * 3  # random mean vec of data
    M = torch.randn(size=(a, d, n))  # random transform matrix of param to data space vec
    C = torch.eye(d)  # cov matrix of dataset
    # mu_theta = torch.randn(size=(1, n))  # random mean vec of parameter
    mu_theta = torch.randn(size=(a, n)) * 3  #
    Sigma = torch.eye(n)  # cov matrix of parameter prior
    sim = Simulator(nreSettings=nreSettings, mu_theta=mu_theta, M=M, mu_data=mu_data, Sigma=Sigma, C=C)

    nreSettings.model = sim.model  # lsbi model
    # generate training dat and obs
    obs = swyft.Sample(x=torch.tensor(sim.model.evidence().rvs()[None, :]))
    training_samples = torch.as_tensor(
        sim.sample(nreSettings.n_training_samples, targets=[nreSettings.targetKey])[
            nreSettings.targetKey])
    comm_gen.Barrier()

    #### instantiate swyft network
    network = Network(nreSettings=nreSettings, obs=obs)

    #### create callbacks function for pytorch lightning trainer
    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=nreSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]

    #### set up polychord settings
    polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
    polyset.file_root = "samples"
    polyset.base_dir = nreSettings.root
    polyset.seed = nreSettings.seed
    polyset.nfail = nreSettings.nlives_per_dim_constant * nreSettings.n_prior_sampling
    polyset.nprior = nreSettings.n_prior_sampling
    polySwyft = PolySwyft(nreSettings=nreSettings, sim=sim, obs=obs, training_samples=training_samples,
                          network=network, polyset=polyset, callbacks=create_callbacks)

    if not nreSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()
    else:
        # load data for plotting if data is already generated
        root_storage, network_storage = reload_data_for_plotting(nreSettings=nreSettings, network=network)
        polySwyft.root_storage = root_storage
        polySwyft.network_storage = network_storage

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(nreSettings=nreSettings, network_storage=polySwyft.network_storage,
                               root_storage=polySwyft.root_storage,
                               sim=sim, obs=obs, polyset=polyset)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
