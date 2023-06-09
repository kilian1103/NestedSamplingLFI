import logging

import numpy as np
import torch
from mpi4py import MPI

from NSLFI.NRE_Post_Analysis import plot_NRE_posterior
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NSNRE_cycle import execute_NSNRE_cycle
from NSLFI.NSNRE_data_generation import DataEnvironment


def execute():
    # add different seed for each rank
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    nreSettings = NRE_Settings()
    np.random.seed(nreSettings.seed)
    torch.manual_seed(nreSettings.seed)
    logging.basicConfig(filename=nreSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    network_storage = dict()
    root_storage = dict()
    # TODO merge prior framework, so far simulator has scipy, polychord has hypercube and MCMC has torch
    dataEnv = DataEnvironment(nreSettings=nreSettings, logger=logger)
    dataEnv.generate_data()
    # retrain NRE and sample new samples with NS loop
    execute_NSNRE_cycle(nreSettings=nreSettings, logger=logger, prior=dataEnv.prior,
                        obs=dataEnv.obs, sim=dataEnv.sim,
                        network_storage=network_storage,
                        root_storage=root_storage, samples=dataEnv.samples,
                        root=dataEnv.root)
    # plot triangle plot
    if rank_gen == 0:
        plot_NRE_posterior(nreSettings=nreSettings, root_storage=root_storage)
        # TODO fix counting code
        # plot_NRE_expansion_and_contraction_rate(nreSettings=nreSettings, root_storage=root_storage)


if __name__ == '__main__':
    execute()
