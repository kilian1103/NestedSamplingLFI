import logging

import anesthetic
import matplotlib.pyplot as plt
import pypolychord
import swyft
import torch
import wandb
from mpi4py import MPI
from swyft import Simulator

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Post_Analysis import plot_quantile_plot
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.utils import compute_KL_divergence


def execute_NSNRE_cycle(nreSettings: NRE_Settings, sim: Simulator,
                        obs: swyft.Sample, network_storage: dict, root_storage: dict, samples: torch.Tensor):
    # retrain NRE and sample new samples with NS loop
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    logger = logging.getLogger(nreSettings.logger_name)
    dkl_storage = list()
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        root = f"{nreSettings.root}_round_{rd}"
        if rank_gen == 0:
            logger.info("retraining round: " + str(rd))
            if nreSettings.activate_wandb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
            network = retrain_next_round(root=root, nextRoundPoints=samples,
                                         nreSettings=nreSettings, sim=sim, obs=obs)
        else:
            network = Network(nreSettings=nreSettings)
        comm_gen.Barrier()
        # load saved network
        network.load_state_dict(torch.load(f"{root}/{nreSettings.neural_network_file}"))
        network.double()  # change to float64 precision of network
        trained_NRE = NRE_PolyChord(network=network, obs=obs)
        network_storage[f"round_{rd}"] = trained_NRE
        root_storage[f"round_{rd}"] = root
        logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")
        # start polychord section
        polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
        polyset.file_root = nreSettings.file_root
        polyset.base_dir = root
        polyset.seed = nreSettings.seed
        polyset.nfail = nreSettings.nlive_scan_run_per_feature * nreSettings.n_training_samples
        polyset.nprior = nreSettings.n_training_samples
        polyset.nlive = nreSettings.nlive_scan_run_per_feature * nreSettings.num_features
        # Run PolyChord
        pypolychord.run_polychord(loglikelihood=trained_NRE.logLikelihood, nDims=nreSettings.num_features,
                                  nDerived=nreSettings.nderived, settings=polyset,
                                  prior=trained_NRE.prior, dumper=trained_NRE.dumper)
        comm_gen.Barrier()
        deadpoints = anesthetic.read_chains(root=f"{root}/{nreSettings.file_root}")
        if rd >= 1:
            DKL = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                        current_samples=deadpoints, rd=rd)
            dkl_storage.append(DKL)
            logger.info(f"DKL of rd {rd} is: {DKL}")
        comm_gen.Barrier()
        deadpoints = deadpoints.iloc[:, :nreSettings.num_features]
        if rank_gen == 0:
            plot_quantile_plot(samples=deadpoints.copy(), percentiles=nreSettings.percentiles_of_quantile_plot,
                               nreSettings=nreSettings,
                               root=root)
        comm_gen.Barrier()
        deadpoints = torch.as_tensor(deadpoints.to_numpy())
        logger.info(f"total data size for training for rd {rd + 1}: {deadpoints.shape[0]}")
        samples = deadpoints

    ### plot KL divergence plot
    plt.figure()
    plt.errorbar(x=[x for x in range(1, nreSettings.NRE_num_retrain_rounds + 1)], y=[x[0] for x in dkl_storage],
                 yerr=[x[1] for x in dkl_storage])
    plt.xlabel("round")
    plt.ylabel("KL divergence")
    plt.savefig(f"{root_storage['round_0']}/kl_divergence.pdf")
