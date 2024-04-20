import logging
import os
from typing import Callable

import anesthetic
import numpy as np
import pandas as pd
import pypolychord
import swyft
import torch
import wandb
from mpi4py import MPI
from pypolychord import PolyChordSettings

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.utils import compute_KL_divergence, select_weighted_contour


class PolySwyft:
    def __init__(self, nreSettings: NRE_Settings, sim: swyft.Simulator,
                 obs: swyft.Sample, deadpoints: torch.Tensor,
                 network: swyft.SwyftModule, polyset: PolyChordSettings,
                 callbacks: Callable):
        self.nreSettings = nreSettings
        self.polyset = polyset
        self.sim = sim
        self.obs = obs
        self.callbacks = callbacks
        self.current_deadpoints = deadpoints
        self.network_model = network
        self.network_storage = dict()
        self.root_storage = dict()
        self.dkl_storage = list()
        self.deadpoints_storage = dict()

    def execute_NSNRE_cycle(self):
        # retrain NRE and sample new samples with NS loop
        self.logger = logging.getLogger(self.nreSettings.logger_name)
        if self.nreSettings.NRE_start_from_round > 0:
            if (self.nreSettings.NRE_start_from_round > self.nreSettings.NRE_num_retrain_rounds and
                    self.nreSettings.cyclic_rounds):
                raise ValueError("NRE_start_from_round must be smaller than NRE_num_retrain_rounds")
            ### only execute this code when previous rounds are already trained ###
            for rd in range(0, self.nreSettings.NRE_start_from_round):
                root = f"{self.nreSettings.root}_round_{rd}"
                network = self.network_model.get_new_network()
                network.double()  # change to float64 precision of network
                network.load_state_dict(torch.load(f"{root}/{self.nreSettings.neural_network_file}"))
                network.eval()
                self.network_storage[rd] = network
                self.root_storage[rd] = root
                if self.nreSettings.use_livepoint_increasing:
                    deadpoints = anesthetic.read_chains(
                        root=f"{root}/{self.nreSettings.increased_livepoints_fileroot}/{self.polyset.file_root}")
                else:
                    deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")  #

                self.deadpoints_storage[rd] = deadpoints.copy()

                if rd > 0:
                    previous_network = self.network_storage[rd - 1]
                    DKL = compute_KL_divergence(nreSettings=self.nreSettings, previous_network=previous_network.eval(),
                                                current_samples=self.deadpoints_storage[rd], obs=self.obs,
                                                previous_samples=self.deadpoints_storage[rd - 1])
                    self.dkl_storage.append(DKL)

                    del self.deadpoints_storage[rd - 1]  # save memory
                    del self.network_storage[rd - 1]  # save memory

            if self.nreSettings.use_dataset_clipping:
                # TODO make non-random seeding compatible
                logR_cutoff = float(self.nreSettings.dataset_logR_cutoff_sigma * deadpoints["logL"].std())
                rest = deadpoints[deadpoints.logL >= logR_cutoff]
                bools = np.random.choice([True, False], size=rest.shape[0],
                                         p=[self.nreSettings.dataset_uniform_sampling_rate,
                                            1 - self.nreSettings.dataset_uniform_sampling_rate])
                rest = rest[bools]
                deadpoints = deadpoints.truncate(logR_cutoff)
                deadpoints = pd.concat([deadpoints, rest], axis=0)
                deadpoints.drop_duplicates(inplace=True)

            deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            self.current_deadpoints = deadpoints

        ### main cycle ###
        if self.nreSettings.cyclic_rounds:
            self._cyclic_rounds()
        else:
            self._cyclic_kl()

        del self.deadpoints_storage
        del self.network_storage

    def _cyclic_rounds(self):
        DKL = 10
        for rd in range(self.nreSettings.NRE_start_from_round, self.nreSettings.NRE_num_retrain_rounds + 1):
            _ = self._cycle(DKL, rd)

    def _cyclic_kl(self):
        DKL_info = (10, 10)
        DKL, DKL_std = DKL_info
        rd = self.nreSettings.NRE_start_from_round
        while abs(DKL) >= self.nreSettings.termination_abs_dkl:
            DKL, DKL_std = self._cycle(DKL_info, rd)
            rd += 1
        self.nreSettings.NRE_num_retrain_rounds = rd - 1

    def _cycle(self, DKL, rd):
        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()

        ### start NRE training section ###
        root = f"{self.nreSettings.root}_round_{rd}"
        self.logger.info("retraining round: " + str(rd))
        if self.nreSettings.activate_wandb and rank_gen == 0:
            try:
                self.finish_kwargs = self.nreSettings.wandb_kwargs.pop("finish")
            except KeyError:
                self.finish_kwargs = {'exit_code': None,
                                      'quiet': None}
            self.nreSettings.wandb_kwargs["name"] = f"round_{rd}"
            wandb.init(**self.nreSettings.wandb_kwargs)

        self.nreSettings.trainer_kwargs["default_root_dir"] = root
        self.nreSettings.trainer_kwargs["callbacks"] = self.callbacks()
        trainer = swyft.SwyftTrainer(**self.nreSettings.trainer_kwargs)
        network = self.network_model.get_new_network()
        network = comm_gen.bcast(network, root=0)

        network = retrain_next_round(root=root, training_data=self.current_deadpoints,
                                     nreSettings=self.nreSettings, sim=self.sim, obs=self.obs,
                                     network=network,
                                     trainer=trainer, rd=rd)
        if self.nreSettings.activate_wandb and rank_gen == 0:
            wandb.finish(**self.finish_kwargs)
        if rank_gen == 0:
            torch.save(network.state_dict(), f"{root}/{self.nreSettings.neural_network_file}")
        comm_gen.Barrier()
        network.eval()

        self.network_storage[rd] = network
        self.root_storage[rd] = root
        self.logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")

        ### start polychord section ###
        ### Run PolyChord ###
        self.polyset.base_dir = root
        self.polyset.nlive = self.nreSettings.nlives_per_round[rd]

        comm_gen.barrier()
        pypolychord.run_polychord(loglikelihood=network.logLikelihood,
                                  nDims=self.nreSettings.num_features,
                                  nDerived=self.nreSettings.nderived, settings=self.polyset,
                                  prior=network.prior, dumper=network.dumper)
        comm_gen.Barrier()

        ### load deadpoints and compute KL divergence and reassign to training samples ###
        deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
        comm_gen.Barrier()
        if self.nreSettings.use_livepoint_increasing:
            index = select_weighted_contour(deadpoints,
                                            threshold=1 - self.nreSettings.livepoint_increase_posterior_contour)
            logL = deadpoints.iloc[index, :].logL

            try:
                os.makedirs(f"{root}/{self.nreSettings.increased_livepoints_fileroot}")
            except OSError:
                self.logger.info("root folder already exists!")

            self.polyset.base_dir = f"{root}/{self.nreSettings.increased_livepoints_fileroot}"
            self.polyset.nlives = {logL: self.nreSettings.n_increased_livepoints}
            comm_gen.Barrier()
            pypolychord.run_polychord(loglikelihood=network.logLikelihood,
                                      nDims=self.nreSettings.num_features,
                                      nDerived=self.nreSettings.nderived, settings=self.polyset,
                                      prior=network.prior, dumper=network.dumper)
            comm_gen.Barrier()
            self.polyset.nlives = {}
            deadpoints = anesthetic.read_chains(
                root=f"{root}/{self.nreSettings.increased_livepoints_fileroot}/{self.polyset.file_root}")
            comm_gen.Barrier()

        self.deadpoints_storage[rd] = deadpoints.copy()

        if rd > 0:
            previous_network = self.network_storage[rd - 1]
            DKL = compute_KL_divergence(nreSettings=self.nreSettings, previous_network=previous_network.eval(),
                                        current_samples=self.deadpoints_storage[rd], obs=self.obs,
                                        previous_samples=self.deadpoints_storage[rd - 1])
            self.dkl_storage.append(DKL)
            self.logger.info(f"DKL of rd {rd} is: {DKL}")

            del self.deadpoints_storage[rd - 1]  # save memory
            del self.network_storage[rd - 1]  # save memory

        if self.nreSettings.use_dataset_clipping:
            logR_cutoff = float(self.nreSettings.dataset_logR_cutoff_sigma * deadpoints["logL"].std())
            rest = deadpoints[deadpoints.logL >= logR_cutoff]
            bools = np.random.choice([True, False], size=rest.shape[0],
                                     p=[self.nreSettings.dataset_uniform_sampling_rate,
                                        1 - self.nreSettings.dataset_uniform_sampling_rate])
            rest = rest[bools]
            deadpoints = deadpoints.truncate(logR_cutoff)
            deadpoints = pd.concat([deadpoints, rest], axis=0)
            deadpoints.drop_duplicates(inplace=True)

        comm_gen.Barrier()
        deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
        deadpoints = torch.as_tensor(deadpoints.to_numpy())
        self.logger.info(f"total data size for training for rd {rd + 1}: {deadpoints.shape[0]}")
        self.current_deadpoints = deadpoints
        return DKL
