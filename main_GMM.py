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
    n_per_core = nreSettings.n_training_samples // size_gen
    if rank_gen == 0:
        n_per_core += nreSettings.n_training_samples % size_gen
    deapoints = sim.sample(n_per_core, targets=[nreSettings.targetKey])[
        nreSettings.targetKey]
    comm_gen.Barrier()
    deapoints = comm_gen.allgather(deapoints)
    deapoints = np.concatenate(deapoints, axis=0)
    comm_gen.Barrier()

    ### generate true posterior for comparison
    cond = {nreSettings.obsKey: obs[nreSettings.obsKey].numpy().squeeze()}
    full_joint = sim.sample(nreSettings.n_weighted_samples, conditions=cond)
    true_logratios = torch.as_tensor(full_joint[nreSettings.contourKey])
    posterior = full_joint[nreSettings.posteriorsKey]
    weights = np.ones(shape=len(posterior))  # direct samples from posterior have weights 1
    params_labels = {i: rf"${nreSettings.targetKey}_{i}$" for i in range(nreSettings.num_features)}
    mcmc_true = MCMCSamples(
        data=posterior, weights=weights.squeeze(),
        logL=true_logratios, labels=params_labels)

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
    polySwyft = PolySwyft(nreSettings=nreSettings, sim=sim, obs=obs, deadpoints=deapoints,
                          network=network, polyset=polyset, callbacks=create_callbacks)

    if not nreSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()

    root_storage, network_storage, samples_storage, dkl_storage = reload_data_for_plotting(nreSettings=nreSettings,
                                                                                           network=network,
                                                                                           polyset=polyset,
                                                                                           until_round=nreSettings.NRE_num_retrain_rounds)

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(nreSettings=nreSettings, network_storage=network_storage,
                               samples_storage=samples_storage, dkl_storage=dkl_storage,
                               obs=obs, true_posterior=mcmc_true, root=nreSettings.root)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
