import logging

import numpy as np
import torch
from mpi4py import MPI

from NSLFI.NRE_Post_Analysis import plot_analysis_of_NSNRE
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NSNRE_cycle import execute_NSNRE_cycle
from NSLFI.NSNRE_data_generation import DataEnvironment
from NSLFI.utils import reload_data_for_plotting


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
    nreSettings.logger = logger
    logger.info('Started')
    network_storage = dict()
    root_storage = dict()
    # TODO merge prior framework, so far simulator has scipy, polychord has hypercube
    dataEnv = DataEnvironment(nreSettings=nreSettings)
    dataEnv.generate_data()
    if not nreSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        execute_NSNRE_cycle(nreSettings=nreSettings,
                            obs=dataEnv.obs, sim=dataEnv.sim,
                            network_storage=network_storage,
                            root_storage=root_storage, training_samples=dataEnv.samples)
    else:
        # load data for plotting if data is already generated
        root_storage, network_storage = reload_data_for_plotting(nreSettings=nreSettings, dataEnv=dataEnv)

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(nreSettings=nreSettings, network_storage=network_storage, root_storage=root_storage,
                               dataEnv=dataEnv)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
