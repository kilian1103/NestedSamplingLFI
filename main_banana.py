import logging
import os

import numpy as np
import swyft
import torch
from mpi4py import MPI

import wandb
from NSLFI.NRE_Intersector import intersect_samples
from NSLFI.NRE_NS_Wrapper import NRE
from NSLFI.NRE_Network import Network
from NSLFI.NRE_Post_Analysis import plot_NRE_posterior, plot_NRE_expansion_and_contraction_rate
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator
from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.NestedSamplerBounds import NestedSamplerBounds


def execute():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    # add different seed for each rank
    np.random.seed(234 + rank_gen)
    torch.manual_seed(234 + rank_gen)
    nreSettings = NRE_Settings()
    logging.basicConfig(filename=nreSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    root = nreSettings.root
    if rank_gen == 0:
        try:
            os.makedirs(root)
        except OSError:
            logger.info("root folder already exists!")
    network_storage = dict()
    root_storage = dict()
    # uniform prior for theta_i
    theta_prior = torch.distributions.uniform.Uniform(low=nreSettings.sim_prior_lower,
                                                      high=nreSettings.sim_prior_lower + nreSettings.sim_prior_upper)
    # wrap prior for NS sampling procedure
    prior = {f"theta_{i}": theta_prior for i in range(nreSettings.num_features)}

    # observation for simulator
    obs = swyft.Sample(x=np.array(nreSettings.num_features * [0]))
    # define forward model settings
    bimodal = False
    sim = Simulator(bounds_z=None, bimodal=bimodal, nreSettings=nreSettings)
    # generate samples using simulator
    if rank_gen == 0:
        samples = torch.as_tensor(
            sim.sample(nreSettings.n_training_samples, targets=[nreSettings.targetKey])[nreSettings.targetKey])
    else:
        samples = torch.empty((nreSettings.n_training_samples, nreSettings.num_features))
    # broadcast samples to all ranks
    samples = comm_gen.bcast(samples, root=0)
    comm_gen.Barrier()
    # retrain NRE and sample new samples with NS loop
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        if rank_gen == 0:
            logger.info("retraining round: " + str(rd))
            if nreSettings.activate_wandb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project=nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
            network = retrain_next_round(root=root, nextRoundPoints=samples,
                                         nreSettings=nreSettings, sim=sim,
                                         prior=prior, obs=obs)
        else:
            network = Network(nreSettings=nreSettings)
        network = comm_gen.bcast(network, root=0)
        comm_gen.Barrier()
        trained_NRE = NRE(network=network, obs=obs)
        network_storage[f"round_{rd}"] = trained_NRE
        root_storage[f"round_{rd}"] = root
        logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")
        with torch.no_grad():
            # generate samples within median contour of prior trained NRE
            loglikes = trained_NRE.logLikelihood(samples)
            median_logL, idx = torch.median(loglikes, dim=-1)
            boundarySample = samples[idx]
            if rank_gen == 0:
                n1 = loglikes[loglikes > median_logL]
                n2 = loglikes[loglikes < median_logL]
                compression = len(n1) / (len(n1) + len(n2))
                logger.info(f"Median compression due to selecting new boundary sample: {compression}")
                torch.save(boundarySample, f"{root}/boundary_sample")
            comm_gen.Barrier()
            nestedSampler = NestedSamplerBounds(logLikelihood=trained_NRE.logLikelihood, livepoints=samples,
                                                prior=prior, root=root, samplertype=nreSettings.ns_sampler,
                                                logLs=loglikes)
            output = nestedSampler.nested_sampling(stop_criterion=nreSettings.ns_stopping_criterion,
                                                   nsamples=nreSettings.n_training_samples,
                                                   keep_chain=nreSettings.ns_keep_chain,
                                                   boundarySample=boundarySample)
            comm_gen.Barrier()
            if rd >= 1 and rank_gen == 0:
                current_samples = torch.load(f"{root}/posterior_samples")
                previous_NRE = network_storage[f"round_{rd - 1}"]
                current_boundary_logL_previous_NRE = previous_NRE.logLikelihood(boundarySample)
                previous_logL_previous_NRE = previous_NRE.logLikelihood(samples)
                n1 = samples[previous_logL_previous_NRE > current_boundary_logL_previous_NRE]
                n2 = samples[previous_logL_previous_NRE < current_boundary_logL_previous_NRE]
                previous_compression_with_current_boundary = len(n1) / (len(n1) + len(n2))
                logger.info(
                    f"Compression of previous NRE contour due to current boundary sample: "
                    f"{previous_compression_with_current_boundary}")
                k1, l1, k2, l2 = intersect_samples(nreSettings=nreSettings, root_storage=root_storage,
                                                   network_storage=network_storage, rd=rd,
                                                   boundarySample=boundarySample,
                                                   current_samples=current_samples,
                                                   previous_samples=n1)
        nextSamples = torch.load(f=f"{root}/posterior_samples")
        newRoot = root + f"_rd_{rd + 1}"
        root = newRoot
        samples = nextSamples
    # plot triangle plot
    if rank_gen == 0:
        plot_NRE_posterior(nreSettings=nreSettings, root_storage=root_storage)
        plot_NRE_expansion_and_contraction_rate(nreSettings=nreSettings, root_storage=root_storage)


if __name__ == '__main__':
    execute()
