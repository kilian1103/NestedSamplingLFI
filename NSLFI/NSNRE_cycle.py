import logging

import numpy as np
import pypolychord
import swyft
import torch
import wandb
from mpi4py import MPI
from swyft import Simulator

from NSLFI.NRE_Intersector import intersect_samples
from NSLFI.NRE_Network import Network
from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_retrain import retrain_next_round


def execute_NSNRE_cycle(nreSettings: NRE_Settings, logger: logging.Logger, sim: Simulator,
                        obs: swyft.Sample, network_storage: dict, root_storage: dict, samples: torch.Tensor, root: str):
    # retrain NRE and sample new samples with NS loop
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    # full_samples = samples.clone()
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        if rank_gen == 0:
            logger.info("retraining round: " + str(rd))
            if nreSettings.activate_wandb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
            # replace full_samples with samples
            network = retrain_next_round(root=root, nextRoundPoints=samples,
                                         nreSettings=nreSettings, sim=sim, obs=obs)
        else:
            network = Network(nreSettings=nreSettings)
        network = comm_gen.bcast(network, root=0)
        comm_gen.Barrier()
        trained_NRE = NRE_PolyChord(network=network, obs=obs)
        network_storage[f"round_{rd}"] = trained_NRE
        root_storage[f"round_{rd}"] = root
        logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")
        # start counting
        if rd >= 1 and nreSettings.activate_NSNRE_counting:
            # TODO fix counting code, polychord has bugs with these settings
            previous_root = root_storage[f"round_{rd - 1}"]
            data = np.loadtxt(f"{previous_root}/{nreSettings.file_root}.txt")
            boundarySample = data[-nreSettings.n_training_samples - 1, 2:]
            logger.info(f"boundarySample: {boundarySample}")
            boundarySample_logL, _ = trained_NRE.logLikelihood(boundarySample)  # recompute ! not reload from file

            livepoint_norm = (boundarySample - nreSettings.sim_prior_lower) / nreSettings.prior_width
            # livepoint_norm = np.tile(livepoint_norm, (size_gen, 1))
            livepoint_norm = livepoint_norm.reshape(1, nreSettings.num_features)
            logger.info(f"livepoints shape: {livepoint_norm.shape}")

            previous_samples = data[-nreSettings.n_training_samples:, 2:]
            comm_gen.Barrier()
            polyset_repop = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
            # repop settings
            polyset_repop.nlive = 1
            polyset_repop.nfail = nreSettings.n_training_samples
            polyset_repop.cube_samples = livepoint_norm
            polyset_repop.nlives = {boundarySample_logL: nreSettings.n_training_samples + 1,
                                    boundarySample_logL + 1e-8: 0}
            # TODO use anesthetic to get livepoints at iteration i, samples
            #  polyset_repop.max_ndead = 1
            # polyset_repop.write_resume = False
            # other settings
            polyset_repop.file_root = "repop"
            polyset_repop.base_dir = root
            polyset_repop.seed = nreSettings.seed
            pypolychord.run_polychord(loglikelihood=trained_NRE.logLikelihood, nDims=nreSettings.num_features,
                                      nDerived=nreSettings.nderived, settings=polyset_repop,
                                      prior=trained_NRE.prior)
            comm_gen.Barrier()
            data = np.loadtxt(f"{root}/{polyset_repop.file_root}.txt")
            samples = data[1:nreSettings.n_training_samples + 1, 2:].reshape(
                nreSettings.n_training_samples,
                nreSettings.num_features)
            if rank_gen == 0:
                k1, l1, k2, l2 = intersect_samples(nreSettings=nreSettings, root_storage=root_storage,
                                                   network_storage=network_storage, rd=rd,
                                                   boundarySample=torch.as_tensor(boundarySample).unsqueeze(0),
                                                   current_samples=torch.as_tensor(samples),
                                                   previous_samples=torch.as_tensor(previous_samples))
        comm_gen.Barrier()
        # start compressing
        samples_norm = (samples - nreSettings.sim_prior_lower) / nreSettings.prior_width
        polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
        polyset.file_root = nreSettings.file_root
        polyset.base_dir = root
        polyset.seed = nreSettings.seed
        polyset.nlive = samples_norm.shape[0]
        polyset.cube_samples = samples_norm
        polyset.max_ndead = int(1 * nreSettings.n_training_samples)  # exp(-max_ndead/n_live) compression
        # Run PolyChord
        pypolychord.run_polychord(loglikelihood=trained_NRE.logLikelihood, nDims=nreSettings.num_features,
                                  nDerived=nreSettings.nderived, settings=polyset,
                                  prior=trained_NRE.prior, dumper=trained_NRE.dumper)
        comm_gen.Barrier()
        # TODO use anesthetc
        nextSamples = np.loadtxt(f"{root}/{nreSettings.file_root}.txt")
        nextSamples = torch.as_tensor(nextSamples[-nreSettings.n_training_samples:, 2:])
        newRoot = root + f"_rd_{rd + 1}"
        root = newRoot
        # full_samples = torch.cat([full_samples, nextSamples], dim=0)
        samples = nextSamples