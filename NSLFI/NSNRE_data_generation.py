import logging
import os

import numpy as np
import swyft
import torch
from mpi4py import MPI

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator


class DataEnvironment:
    def __init__(self, nreSettings: NRE_Settings):
        self.nreSettings = nreSettings
        # define forward model settings
        self.sim = Simulator(bounds_z=None, bimodal=True, nreSettings=self.nreSettings)

    def generate_data(self):
        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()
        logger = logging.getLogger(self.nreSettings.logger_name)
        if rank_gen == 0:
            try:
                os.makedirs(self.nreSettings.root)
            except OSError:
                logger.info("root folder already exists!")

        # observation for simulator
        obs = swyft.Sample(x=np.array(self.nreSettings.num_features * [0]))
        # generate samples using simulator
        if rank_gen == 0:
            samples = torch.as_tensor(
                sim.sample(self.nreSettings.n_training_samples, targets=[self.nreSettings.targetKey])[
                    self.nreSettings.targetKey])
            torch.save(samples, f"{self.nreSettings.root}/full_prior_samples")
        else:
            samples = torch.empty((self.nreSettings.n_training_samples, self.nreSettings.num_features))
        # broadcast samples to all ranks
        samples = comm_gen.bcast(samples, root=0)
        comm_gen.Barrier()
        # save datageneration results
        self.samples = samples
        self.obs = obs
