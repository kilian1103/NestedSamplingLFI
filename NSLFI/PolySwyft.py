import logging
import os
from typing import Callable

import pypolychord
import wandb
from pypolychord import PolyChordSettings

from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.utils import *


class PolySwyft:
    def __init__(self, nreSettings: NRE_Settings, sim: swyft.Simulator,
                 obs: swyft.Sample, deadpoints: torch.Tensor,
                 network: swyft.SwyftModule, polyset: PolyChordSettings,
                 callbacks: Callable):
        """
        Initialize the PolySwyft object.
        :param nreSettings: A NRE_Settings object
        :param sim: A swyft simulator object
        :param obs: A swyft sample of the observed data
        :param deadpoints: A torch.Tensor of the deadpoints
        :param network: A swyft network object
        :param polyset: A PolyChordSettings object
        :param callbacks: A callable object for instantiating the new callbacks of the pl.trainer
        """
        self.nreSettings = nreSettings
        self.polyset = polyset
        self.sim = sim
        self.obs = obs
        self.callbacks = callbacks
        self.current_deadpoints = deadpoints
        self.network_model = network
        self.network_storage = dict()
        self.root_storage = dict()
        self.dkl_storage = dict()
        self.deadpoints_storage = dict()

    def execute_NSNRE_cycle(self):
        """
        Execute the sequential nested sampling neural ratio estimation cycle.
        :return:
        """
        try:
            from mpi4py import MPI
        except ImportError:
            raise ImportError("mpi4py is required for PolySwyft!")
        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()

        self.logger = logging.getLogger(self.nreSettings.logger_name)

        ### reload data if necessary to resume run ###
        if self.nreSettings.NRE_start_from_round > 0:
            if (self.nreSettings.NRE_start_from_round > self.nreSettings.NRE_num_retrain_rounds and
                    self.nreSettings.cyclic_rounds):
                raise ValueError("NRE_start_from_round must be smaller than NRE_num_retrain_rounds")
            self._reload_data()
            deadpoints = self.deadpoints_storage[self.nreSettings.NRE_start_from_round - 1]

            ### truncate last set of deadpoints for resuming training if neccessary ###
            if self.nreSettings.use_dataset_truncation:
                logR_cutoff = float(self.nreSettings.dataset_logR_cutoff)
                deadpoints = deadpoints.truncate(logR_cutoff)
                ### sample random deadpoints within omitted sample###
                if self.nreSettings.use_dataset_random_sampling:
                    p = self.nreSettings.dataset_uniform_sampling_rate
                    if rank_gen == 0:
                        deadpoints = random_subset_after_truncation(deadpoints=deadpoints, logR_cutoff=logR_cutoff, p=p)
                    comm_gen.Barrier()
                    deadpoints = comm_gen.bcast(deadpoints, root=0)

            ### save current deadpoints for next training round ###
            deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            self.current_deadpoints = deadpoints

        ### execute main cycle ###
        if self.nreSettings.cyclic_rounds:
            self._cyclic_rounds()
        else:
            self._cyclic_kl()

        ### delete temporary storage as results are saved on disk ###
        del self.deadpoints_storage
        del self.network_storage

    def _cyclic_rounds(self):
        for rd in range(self.nreSettings.NRE_start_from_round, self.nreSettings.NRE_num_retrain_rounds + 1):
            self._cycle(rd)

    def _cyclic_kl(self):
        DKL_info = (100, 100)
        DKL, DKL_std = DKL_info
        rd = self.nreSettings.NRE_start_from_round
        while abs(DKL) >= self.nreSettings.termination_abs_dkl:
            self._cycle(rd)
            DKL, DKL_std = self.dkl_storage[rd]
            rd += 1
        self.nreSettings.NRE_num_retrain_rounds = rd - 1

    def _cycle(self, rd):
        try:
            from mpi4py import MPI
        except ImportError:
            raise ImportError("mpi4py is required for PolySwyft!")

        comm_gen = MPI.COMM_WORLD
        rank_gen = comm_gen.Get_rank()
        size_gen = comm_gen.Get_size()

        ### start NRE training section ###
        root = f"{self.nreSettings.root}_round_{rd}"
        self.logger.info("training network round: " + str(rd))

        ### setup wandb ###
        if self.nreSettings.activate_wandb and rank_gen == 0:
            try:
                self.finish_kwargs = self.nreSettings.wandb_kwargs.pop("finish")
            except KeyError:
                self.finish_kwargs = {'exit_code': None,
                                      'quiet': None}
            self.nreSettings.wandb_kwargs["name"] = f"round_{rd}"
            wandb.init(**self.nreSettings.wandb_kwargs)

        ### setup trainer ###
        self.nreSettings.trainer_kwargs["default_root_dir"] = root
        self.nreSettings.trainer_kwargs["callbacks"] = self.callbacks()
        trainer = swyft.SwyftTrainer(**self.nreSettings.trainer_kwargs)

        ### setup network and train network###
        if self.nreSettings.active_learning_mode:
            network = self.network_model
        else:
            network = self.network_model.get_new_network()

        network = comm_gen.bcast(network, root=0)
        network = retrain_next_round(root=root, deadpoints=self.current_deadpoints,
                                     nreSettings=self.nreSettings, sim=self.sim,
                                     network=network,
                                     trainer=trainer, rd=rd)

        if self.nreSettings.activate_wandb and rank_gen == 0:
            wandb.finish(**self.finish_kwargs)

        ### save network on disk ###
        if rank_gen == 0:
            torch.save(network.state_dict(), f"{root}/{self.nreSettings.neural_network_file}")

        ### save network and root in memory###
        comm_gen.Barrier()
        network.eval()
        self.network_storage[rd] = network
        self.root_storage[rd] = root

        ### start polychord section ###
        ### run PolyChord ###
        self.logger.info("Using PolyChord with trained NRE to generate deadpoints for the next round!")
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

        ### polychord round 2 section ###
        if self.nreSettings.use_livepoint_increasing:

            ### choose contour to increase livepoints ###
            index = select_weighted_contour(deadpoints,
                                            threshold=1 - self.nreSettings.livepoint_increase_posterior_contour)
            logL = deadpoints.iloc[index, :].logL

            try:
                os.makedirs(f"{root}/{self.nreSettings.increased_livepoints_fileroot}")
            except OSError:
                self.logger.info("root folder already exists!")

            ### run polychord round 2 ###
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

        ### compute KL divergence ###
        if rd > 0:
            previous_network = self.network_storage[rd - 1]
            DKL = compute_KL_divergence(nreSettings=self.nreSettings, previous_network=previous_network.eval(),
                                        current_samples=self.deadpoints_storage[rd], obs=self.obs,
                                        previous_samples=self.deadpoints_storage[rd - 1])
            self.dkl_storage[rd] = DKL
            self.logger.info(f"DKL of rd {rd} is: {DKL}")

            ### delete previous deadpoints and network to save temporary memory, as saved on disk ###
            del self.deadpoints_storage[rd - 1]
            del self.network_storage[rd - 1]

        ### truncate deadpoints ###
        if self.nreSettings.use_dataset_truncation:
            logR_cutoff = float(self.nreSettings.dataset_logR_cutoff)
            deadpoints = deadpoints.truncate(logR_cutoff)
            ### sample random deadpoints within omitted sample###
            if self.nreSettings.use_dataset_random_sampling:
                p = self.nreSettings.dataset_uniform_sampling_rate
                if rank_gen == 0:
                    deadpoints = random_subset_after_truncation(deadpoints=deadpoints, logR_cutoff=logR_cutoff, p=p)
                comm_gen.Barrier()
                deadpoints = comm_gen.bcast(deadpoints, root=0)

        comm_gen.Barrier()

        if self.nreSettings.use_posterior_compression:
            deadpoints = deadpoints.compress()

        ### save current deadpoints for next round ###

        deadpoints = deadpoints.iloc[:, :self.nreSettings.num_features]
        deadpoints = torch.as_tensor(deadpoints.to_numpy())
        self.logger.info(f"Number of deadpoints for next rd {rd + 1}: {deadpoints.shape[0]}")
        self.current_deadpoints = deadpoints
        return

    def _reload_data(self):
        root_storage, network_storage, samples_storage, dkl_storage = reload_data_for_plotting(
            nreSettings=self.nreSettings,
            network=self.network_model,
            polyset=self.polyset,
            until_round=self.nreSettings.NRE_start_from_round - 1,
            only_last_round=True)
        self.root_storage = root_storage
        self.network_storage = network_storage
        self.deadpoints_storage = samples_storage
        self.dkl_storage = dkl_storage

        del self.network_storage[self.nreSettings.NRE_start_from_round - 2]
        del self.deadpoints_storage[self.nreSettings.NRE_start_from_round - 2]
