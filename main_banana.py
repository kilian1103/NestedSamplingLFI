import logging
import os

import numpy as np
import swyft
import torch

import wandb
from NSLFI.NRE_Intersector import intersect_samples
from NSLFI.NRE_NS_Wrapper import NRE
from NSLFI.NRE_Post_Analysis import plot_NRE_posterior
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator
from NSLFI.NRE_retrain import retrain_next_round
from NSLFI.NestedSamplerBounds import NestedSamplerBounds


def execute():
    np.random.seed(234)
    torch.manual_seed(234)
    nreSettings = NRE_Settings()
    logging.basicConfig(filename=nreSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    root = nreSettings.root
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
    samples = torch.as_tensor(
        sim.sample(nreSettings.n_training_samples, targets=[nreSettings.targetKey])[nreSettings.targetKey])

    # retrain NRE and sample new samples with NS loop
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        logger.info("retraining round: " + str(rd))
        wandb.init(
            # set the wandb project where this run will be logged
            project=nreSettings.wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
        network = retrain_next_round(root=root, nextRoundPoints=samples,
                                     nreSettings=nreSettings, sim=sim,
                                     prior=prior, obs=obs)
        wandb.finish()
        network_storage[f"round_{rd}"] = network
        root_storage[f"round_{rd}"] = root

        trained_NRE = NRE(network=network, obs=obs)
        logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")
        with torch.no_grad():
            # rd 0 generate prior samples theta_00, train NRE_0, evaluate med_0, generate new samples theta_0 > med_0
            # rd 1 train NRE_1, generate new samples theta_1 > med_0, evaluate med_1
            # rd 2 train NRE_2, generate new samples theta_2 > med_1, evaluate med_2
            # rd 3 train NRE_3, generate new samples theta_3 > med_2, evaluate med_3
            _, idx = torch.median(trained_NRE.logLikelihood(samples), dim=-1)
            boundarySample = samples[idx]
            torch.save(boundarySample, f"{root}/boundary_sample")
            nestedSampler = NestedSamplerBounds(logLikelihood=trained_NRE.logLikelihood, livepoints=samples,
                                                prior=prior, root=root, samplertype=nreSettings.ns_sampler)
            output = nestedSampler.nested_sampling(stop_criterion=nreSettings.ns_stopping_criterion,
                                                   nsamples=nreSettings.n_training_samples,
                                                   keep_chain=nreSettings.ns_keep_chain,
                                                   boundarySample=boundarySample)

        if rd >= 1:
            intersect_samples(nreSettings=nreSettings, root_storage=root_storage, obs=obs,
                              network_storage=network_storage, rd=rd)

        nextSamples = torch.load(f=f"{root}/posterior_samples")
        newRoot = root + f"_rd_{rd + 1}"
        root = newRoot
        samples = nextSamples
    # plot triangle plot
    plot_NRE_posterior(nreSettings=nreSettings, root_storage=root_storage)


if __name__ == '__main__':
    execute()
