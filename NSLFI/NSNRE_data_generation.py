import logging
import os

import numpy as np
import swyft
import torch
from mpi4py import MPI

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator


class DataEnvironment:
    def __init__(self, nreSettings: NRE_Settings, logger: logging.Logger):
        self.nreSettings = nreSettings
        self.logger = logger

    def generate_data(self):
        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()
        root = self.nreSettings.root
        if rank_gen == 0:
            try:
                os.makedirs(root)
            except OSError:
                self.logger.info("root folder already exists!")
        # uniform prior for theta_i
        theta_prior = torch.distributions.uniform.Uniform(low=self.nreSettings.sim_prior_lower,
                                                          high=self.nreSettings.sim_prior_lower +
                                                               self.nreSettings.prior_width)
        # wrap prior for NS sampling procedure
        prior = {f"theta_{i}": theta_prior for i in range(self.nreSettings.num_features)}

        # observation for simulator
        obs = swyft.Sample(x=np.array(self.nreSettings.num_features * [0]))
        # define forward model settings
        sim = Simulator(bounds_z=None, bimodal=True, nreSettings=self.nreSettings)
        # generate samples using simulator
        if rank_gen == 0:
            samples = torch.as_tensor(
                sim.sample(self.nreSettings.n_training_samples, targets=[self.nreSettings.targetKey])[
                    self.nreSettings.targetKey])
        else:
            samples = torch.empty((self.nreSettings.n_training_samples, self.nreSettings.num_features))
        # broadcast samples to all ranks
        samples = comm_gen.bcast(samples, root=0)
        comm_gen.Barrier()
        # save datageneration results
        self.sim = sim
        self.samples = samples
        self.prior = prior
        self.obs = obs
        self.root = root
