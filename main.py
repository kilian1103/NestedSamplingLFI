import logging

import numpy as np
import swyft
import torch
from mpi4py import MPI

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Post_Analysis import plot_analysis_of_NSNRE
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator_MultiGauss import Simulator
from NSLFI.NSNRE_cycle import execute_NSNRE_cycle
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
    # instantiate swyft simulator
    sim = Simulator(nreSettings=nreSettings)
    # generate training dat aand obs
    obs = swyft.Sample(x=np.array(nreSettings.num_features_dataset * [0]))
    if rank_gen == 0:
        training_samples = torch.as_tensor(
            sim.sample(nreSettings.n_training_samples, targets=[nreSettings.targetKey])[
                nreSettings.targetKey])
    else:
        training_samples = torch.empty((nreSettings.n_training_samples, nreSettings.num_features))
    # broadcast samples to all ranks
    training_samples = comm_gen.bcast(training_samples, root=0)
    comm_gen.Barrier()
    untrained_network = Network(nreSettings=nreSettings)
    if not nreSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        execute_NSNRE_cycle(nreSettings=nreSettings,
                            obs=obs, sim=sim,
                            network_storage=network_storage,
                            root_storage=root_storage, training_samples=training_samples,
                            untrained_network=untrained_network)
    else:
        # load data for plotting if data is already generated
        root_storage, network_storage = reload_data_for_plotting(nreSettings=nreSettings, obs=obs)

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(nreSettings=nreSettings, network_storage=network_storage, root_storage=root_storage,
                               sim=sim, obs=obs)
    logger.info('Finished')


if __name__ == '__main__':
    execute()
