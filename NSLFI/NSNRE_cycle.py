import logging

import anesthetic
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
from NSLFI.utils import random_subset, select_weighted_contour


def execute_NSNRE_cycle(nreSettings: NRE_Settings, sim: Simulator,
                        obs: swyft.Sample, network_storage: dict, root_storage: dict, samples: torch.Tensor, root: str):
    # retrain NRE and sample new samples with NS loop
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    full_samples = samples.clone()
    logger = logging.getLogger(nreSettings.logger_name)
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        if rank_gen == 0:
            logger.info("retraining round: " + str(rd))
            if nreSettings.activate_wandb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
                # thin samples
            if rd == 0:
                # first round, use all samples
                subset = full_samples
            else:
                # thin samples for next round
                subset = random_subset(dataset=full_samples, thinning_factor=nreSettings.thinning_factor_of_dataset)

            network = retrain_next_round(root=root, nextRoundPoints=subset,
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
        # start counting
        if rd >= 1 and nreSettings.activate_NSNRE_counting:
            # TODO fix counting code, polychord has bugs with these settings
            previous_root = root_storage[f"round_{rd - 1}"]
            previous_samples = anesthetic.read_chains(root=f"{previous_root}/{nreSettings.file_root}")

            boundarySample = previous_samples.iloc[-nreSettings.n_training_samples - 1,
                             :nreSettings.num_features].to_numpy()
            boundarySample_logL, _ = trained_NRE.logLikelihood(boundarySample)  # recompute ! not reload from file

            livepoint_norm = (boundarySample - nreSettings.sim_prior_lower) / nreSettings.prior_width
            livepoint_norm = livepoint_norm.reshape(1, nreSettings.num_features)
            previous_samples = previous_samples.live_points().iloc[:, :nreSettings.num_features].to_numpy()
            comm_gen.Barrier()

            # repop settings
            polyset_repop = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
            polyset_repop.nlive = 1
            polyset_repop.nfail = nreSettings.n_training_samples
            polyset_repop.cube_samples = livepoint_norm
            polyset_repop.nlives = {boundarySample_logL: nreSettings.n_training_samples,
                                    boundarySample_logL + 1e-3: 0}
            # other settings
            polyset_repop.file_root = "repop"
            polyset_repop.base_dir = root
            polyset_repop.seed = nreSettings.seed
            pypolychord.run_polychord(loglikelihood=trained_NRE.logLikelihood, nDims=nreSettings.num_features,
                                      nDerived=nreSettings.nderived, settings=polyset_repop,
                                      prior=trained_NRE.prior)
            comm_gen.Barrier()
            data = anesthetic.read_chains(root=f"{root}/{polyset_repop.file_root}")
            samples = data.live_points(1).iloc[:nreSettings.n_training_samples,
                      :nreSettings.num_features].to_numpy().reshape(
                nreSettings.n_training_samples,
                nreSettings.num_features)
            if rank_gen == 0:
                k1, l1, k2, l2 = intersect_samples(nreSettings=nreSettings, root_storage=root_storage,
                                                   network_storage=network_storage, rd=rd,
                                                   boundarySample=torch.as_tensor(boundarySample).unsqueeze(0),
                                                   current_samples=torch.as_tensor(samples),
                                                   previous_samples=torch.as_tensor(previous_samples))
        comm_gen.Barrier()
        # start light run to choose boundary contour
        polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
        polyset.file_root = nreSettings.file_root
        polyset.base_dir = root
        polyset.seed = nreSettings.seed
        polyset.nfail = nreSettings.nlive_scan_run_per_feature * nreSettings.n_training_samples
        if rd >= 1 and not nreSettings.activate_NSNRE_deadpoints_training:
            # thin samples  to get livepoints
            thinning_factor = (nreSettings.nlive_scan_run_per_feature * nreSettings.num_features) / nextSamples.shape[0]
            livepoints = random_subset(dataset=nextSamples, thinning_factor=thinning_factor)
            livepoints_norm = (livepoints - nreSettings.sim_prior_lower) / nreSettings.prior_width
            polyset.cube_samples = livepoints_norm.numpy().reshape(livepoints_norm.shape[0],
                                                                   nreSettings.num_features)
            polyset.nlive = livepoints_norm.shape[0]
        else:
            polyset.nlive = nreSettings.nlive_scan_run_per_feature * nreSettings.num_features
        # Run PolyChord
        pypolychord.run_polychord(loglikelihood=trained_NRE.logLikelihood, nDims=nreSettings.num_features,
                                  nDerived=nreSettings.nderived, settings=polyset,
                                  prior=trained_NRE.prior, dumper=trained_NRE.dumper)
        comm_gen.Barrier()
        scanSamples = anesthetic.read_chains(root=f"{root}/{nreSettings.file_root}")
        boundarySampleIdx = select_weighted_contour(data=scanSamples,
                                                    threshold=nreSettings.anesthetic_sample_threshold)
        boundarySample = scanSamples.iloc[boundarySampleIdx, :nreSettings.num_features].to_numpy()
        boundarySample_logL, _ = trained_NRE.logLikelihood(boundarySample)  # recompute ! not reload from file
        logger.info(f"Boundary sample logL idx: {boundarySampleIdx}")
        logger.info(f"Boundary sample logL: {boundarySample_logL}")

        # full run
        if not nreSettings.activate_NSNRE_deadpoints_training:
            polyset_enhanced = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
            polyset_enhanced.file_root = nreSettings.enhanced_run_file_root
            polyset_enhanced.base_dir = root
            polyset_enhanced.seed = nreSettings.seed
            polyset_enhanced.nfail = nreSettings.nlive_scan_run_per_feature * nreSettings.n_training_samples
            polyset_enhanced.nlives = {
                boundarySample_logL - nreSettings.nlives_logL_coefficient: nreSettings.n_training_samples,
                boundarySample_logL + nreSettings.nlives_logL_coefficient: 0}
            if rd >= 1:
                polyset_enhanced.cube_samples = polyset.cube_samples
                polyset_enhanced.nlive = polyset.cube_samples.shape[0]
            else:
                polyset_enhanced.nlive = nreSettings.nlive_scan_run_per_feature * nreSettings.num_features
            pypolychord.run_polychord(loglikelihood=trained_NRE.logLikelihood, nDims=nreSettings.num_features,
                                      nDerived=nreSettings.nderived, settings=polyset_enhanced,
                                      prior=trained_NRE.prior, dumper=trained_NRE.dumper)
            comm_gen.Barrier()
            nextSamples = anesthetic.read_chains(root=f"{root}/{nreSettings.enhanced_run_file_root}")
            nextSamples = nextSamples.live_points(boundarySample_logL).iloc[:, :nreSettings.num_features]
        else:
            nextSamples = scanSamples.iloc[:, :nreSettings.num_features]
        nextSamples = torch.as_tensor(nextSamples.to_numpy())
        logger.info(
            f"number of samples for next round after polychord dynamic live compression: {nextSamples.shape[0]}")
        root += f"_rd_{rd + 1}"
        full_samples = torch.cat([full_samples, nextSamples.clone()], dim=0)
        logger.info(f"total data size for training: {full_samples.shape[0]}")
