import gc
import logging

import numpy as np
import pypolychord
import swyft
import torch
from anesthetic import MCMCSamples
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from PolySwyft.PolySwyft_Network import Network
from PolySwyft.PolySwyft_Post_Analysis import plot_analysis_of_NSNRE
from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.PolySwyft_Simulator_MultiGauss import Simulator
from PolySwyft.PolySwyft import PolySwyft
from PolySwyft.utils import reload_data_for_plotting


def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    polyswyftSettings = PolySwyft_Settings()
    seed_everything(polyswyftSettings.seed, workers=True)
    logging.basicConfig(filename=polyswyftSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    polyswyftSettings.logger = logger
    logger.info('Started')

    #### instantiate swyft simulator
    d = polyswyftSettings.num_features_dataset
    n = polyswyftSettings.num_features

    m = torch.randn(d) * 3  # mean vec of dataset
    M = torch.randn(size=(d, n))  # transform matrix of dataset to parameter vee
    C = torch.eye(d)  # cov matrix of dataset
    # C very small, or Sigma very big
    mu = torch.zeros(n)  # mean vec of parameter prior
    Sigma = 100 * torch.eye(n)  # cov matrix of parameter prior
    sim = Simulator(polyswyftSettings=polyswyftSettings, m=m, M=M, C=C, mu=mu, Sigma=Sigma)
    polyswyftSettings.model = sim.model  # lsbi model

    # generate training dat and obs
    obs = swyft.Sample(x=torch.tensor(sim.model.evidence().rvs()[None, :]))
    n_per_core = polyswyftSettings.n_training_samples // size_gen
    if rank_gen == 0:
        n_per_core += polyswyftSettings.n_training_samples % size_gen
    seed_everything(polyswyftSettings.seed + rank_gen, workers=True)
    deadpoints = sim.sample(n_per_core, targets=[polyswyftSettings.targetKey])[
        polyswyftSettings.targetKey]
    comm_gen.Barrier()
    seed_everything(polyswyftSettings.seed, workers=True)
    deadpoints = comm_gen.allgather(deadpoints)
    deadpoints = np.concatenate(deadpoints, axis=0)
    comm_gen.Barrier()

    ### generate true posterior for comparison
    cond = {polyswyftSettings.obsKey: obs[polyswyftSettings.obsKey].numpy().squeeze()}
    full_joint = sim.sample(polyswyftSettings.n_weighted_samples, conditions=cond)
    true_logratios = torch.as_tensor(full_joint[polyswyftSettings.contourKey])
    posterior = full_joint[polyswyftSettings.posteriorsKey]
    weights = np.ones(shape=len(posterior))  # direct samples from posterior have weights 1
    params_labels = {i: rf"${polyswyftSettings.targetKey}_{i}$" for i in range(polyswyftSettings.num_features)}
    mcmc_true = MCMCSamples(
        data=posterior, weights=weights.squeeze(),
        logL=true_logratios, labels=params_labels)

    #### instantiate swyft networ
    network = Network(polyswyftSettings=polyswyftSettings, obs=obs)

    #### create callbacks function for pytorch lightning trainer
    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=polyswyftSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]

    def lr_round_scheduler(rd: int)-> float:
        lr = polyswyftSettings.learning_rate_init * (polyswyftSettings.learning_rate_decay ** (polyswyftSettings.early_stopping_patience*rd))
        return lr

    #### set up polychord settings
    polyset = pypolychord.PolyChordSettings(polyswyftSettings.num_features, nDerived=polyswyftSettings.nderived)
    polyset.file_root = "samples"
    polyset.base_dir = polyswyftSettings.root
    polyset.seed = polyswyftSettings.seed
    polyset.nfail = polyswyftSettings.n_training_samples
    polyset.nlive = 100*polyswyftSettings.num_features
    polySwyft = PolySwyft(polyswyftSettings=polyswyftSettings, sim=sim, obs=obs, deadpoints=deadpoints,
                          network=network, polyset=polyset, callbacks=create_callbacks, lr_round_scheduler=lr_round_scheduler)
    del deadpoints
    if not polyswyftSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()

    root_storage, network_storage, samples_storage, dkl_storage = reload_data_for_plotting(
        polyswyftSettings=polyswyftSettings,
        network=network,
        polyset=polyset,
        until_round=polyswyftSettings.NRE_num_retrain_rounds)

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(polyswyftSettings=polyswyftSettings, network_storage=network_storage,
                               samples_storage=samples_storage, dkl_storage=dkl_storage,
                               obs=obs, true_posterior=mcmc_true, root=polyswyftSettings.root)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
