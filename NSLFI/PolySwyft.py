import logging
import os
from typing import Callable

import anesthetic
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
                 obs: swyft.Sample, training_samples: torch.Tensor,
                 network: swyft.SwyftModule, polyset: PolyChordSettings,
                 callbacks: Callable):
        self.nreSettings = nreSettings
        self.polyset = polyset
        self.sim = sim
        self.obs = obs
        self.callbacks = callbacks
        self.training_samples = training_samples
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
                    deadpoints = anesthetic.read_chains(root=f"{root}/{self.polyset.file_root}")
                if self.nreSettings.use_dataset_clipping:
                    index = select_weighted_contour(deadpoints, 1 - self.nreSettings.dataset_posterior_clipping_contour)
                    deadpoints = deadpoints.truncate(index)
                self.deadpoints_storage[rd] = deadpoints
                if rd > 0:
                    previous_network = self.network_storage[rd - 1]
                    DKL = compute_KL_divergence(nreSettings=self.nreSettings, previous_network=previous_network.eval(),
                                                current_samples=deadpoints.copy(), obs=self.obs,
                                                previous_samples=self.deadpoints_storage[rd - 1])
                    self.dkl_storage.append(DKL)

            deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            self.training_samples = deadpoints

        ### main cycle ###
        if self.nreSettings.cyclic_rounds:
            self._cyclic_rounds()
        else:
            self._cyclic_kl()

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
        if self.nreSettings.activate_wandb:
            self.nreSettings.wandb_kwargs["group"] = root
            self.nreSettings.wandb_kwargs["name"] = f"round_{rd}"
            wandb.init(**self.nreSettings.wandb_kwargs)
        self.nreSettings.trainer_kwargs["default_root_dir"] = root
        self.nreSettings.trainer_kwargs["callbacks"] = self.callbacks()
        trainer = swyft.SwyftTrainer(**self.nreSettings.trainer_kwargs)
        network = self.network_model.get_new_network()
        network = retrain_next_round(root=root, training_data=self.training_samples,
                                     nreSettings=self.nreSettings, sim=self.sim, obs=self.obs,
                                     network=network,
                                     trainer=trainer, rd=rd)
        if self.nreSettings.activate_wandb:
            wandb.finish(**self.nreSettings.wandb_finish_kwargs)
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
        if self.nreSettings.use_dataset_clipping:
            index = select_weighted_contour(deadpoints,
                                            threshold=1 - self.nreSettings.dataset_posterior_clipping_contour)
            deadpoints = deadpoints.truncate(index)
            comm_gen.Barrier()

        self.deadpoints_storage[rd] = deadpoints
        if rd >= 1:
            previous_network = self.network_storage[rd - 1]
            DKL = compute_KL_divergence(nreSettings=self.nreSettings, previous_network=previous_network.eval(),
                                        current_samples=deadpoints, obs=self.obs,
                                        previous_samples=self.deadpoints_storage[rd - 1])
            self.dkl_storage.append(DKL)
            self.logger.info(f"DKL of rd {rd} is: {DKL}")
        comm_gen.Barrier()
        deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
        deadpoints = torch.as_tensor(deadpoints.to_numpy())
        self.logger.info(f"total data size for training for rd {rd + 1}: {deadpoints.shape[0]}")
        self.training_samples = deadpoints
        return DKL
