import logging

import anesthetic
import pypolychord
import swyft
import torch
import wandb
from mpi4py import MPI

from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.utils import compute_KL_divergence


def execute_NSNRE_cycle(nreSettings: NRE_Settings, sim: swyft.Simulator,
                        obs: swyft.Sample, network_storage: dict, root_storage: dict, training_samples: torch.Tensor,
                        untrained_network: swyft.SwyftModule):
    # retrain NRE and sample new samples with NS loop
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    logger = logging.getLogger(nreSettings.logger_name)
    dkl_storage = list()

    if nreSettings.NRE_start_from_round > 0:
        if nreSettings.NRE_start_from_round > nreSettings.NRE_num_retrain_rounds:
            raise ValueError("NRE_start_from_round must be smaller than NRE_num_retrain_rounds")
        ### only execute this code when previous rounds are already trained ###
        for i in range(0, nreSettings.NRE_start_from_round):
            root = f"{nreSettings.root}_round_{i}"
            current_network = untrained_network.get_new_network()
            current_network.load_state_dict(torch.load(f"{root}/{nreSettings.neural_network_file}"))
            current_network.double()  # change to float64 precision of network
            trained_NRE = NRE_PolyChord(network=current_network, obs=obs)
            network_storage[f"round_{i}"] = trained_NRE
            root_storage[f"round_{i}"] = root
            if i > 0:
                deadpoints = anesthetic.read_chains(root=f"{root}/{nreSettings.file_root}")
                DKL = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                            current_samples=deadpoints.copy(), rd=i)
                dkl_storage.append(DKL)
        deadpoints = deadpoints.iloc[:, :nreSettings.num_features]
        deadpoints = torch.as_tensor(deadpoints.to_numpy())
        training_samples = deadpoints

    ### main cycle ###
    for rd in range(nreSettings.NRE_start_from_round, nreSettings.NRE_num_retrain_rounds + 1):

        ### start NRE training section ###
        root = f"{nreSettings.root}_round_{rd}"
        if rank_gen == 0:
            logger.info("retraining round: " + str(rd))
            if nreSettings.activate_wandb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
            network = retrain_next_round(root=root, training_data=training_samples,
                                         nreSettings=nreSettings, sim=sim, obs=obs, untrained_network=untrained_network)
        else:
            network = untrained_network.get_new_network()
        comm_gen.Barrier()
        ### load saved network and save it in network_storage ###
        network.load_state_dict(torch.load(f"{root}/{nreSettings.neural_network_file}"))
        network.double()  # change to float64 precision of network
        trained_NRE = NRE_PolyChord(network=network, obs=obs)
        network_storage[f"round_{rd}"] = trained_NRE
        root_storage[f"round_{rd}"] = root
        logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")

        ### start polychord section ###
        polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
        polyset.file_root = nreSettings.file_root
        polyset.base_dir = root
        polyset.seed = nreSettings.seed
        polyset.nfail = nreSettings.nlive_scan_run_per_feature * nreSettings.n_training_samples
        polyset.nprior = nreSettings.n_training_samples
        polyset.nlive = nreSettings.nlive_scan_run_per_feature * nreSettings.num_features
        ### Run PolyChord ###
        pypolychord.run_polychord(loglikelihood=trained_NRE.logLikelihood, nDims=nreSettings.num_features,
                                  nDerived=nreSettings.nderived, settings=polyset,
                                  prior=trained_NRE.prior, dumper=trained_NRE.dumper)
        comm_gen.Barrier()

        ### load deadpoints and compute KL divergence and reassign to training samples ###
        deadpoints = anesthetic.read_chains(root=f"{root}/{nreSettings.file_root}")
        if rd >= 1:
            DKL = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                        current_samples=deadpoints, rd=rd)
            dkl_storage.append(DKL)
            logger.info(f"DKL of rd {rd} is: {DKL}")
        comm_gen.Barrier()
        deadpoints = deadpoints.iloc[:, :nreSettings.num_features]
        deadpoints = torch.as_tensor(deadpoints.to_numpy())
        logger.info(f"total data size for training for rd {rd + 1}: {deadpoints.shape[0]}")
        training_samples = deadpoints
