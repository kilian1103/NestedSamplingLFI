import logging
import os

import numpy as np
import swyft
import torch

import wandb
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator
from NSLFI.NRE_retrain import retrain_next_round_and_generate_new_samples


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
        network = retrain_next_round_and_generate_new_samples(root=root, nextRoundPoints=samples,
                                                              nreSettings=nreSettings, sim=sim,
                                                              prior=prior, obs=obs)
        network_storage[f"round_{rd}"] = network
        root_storage[f"round_{rd}"] = root
        wandb.finish()
        nextSamples = torch.load(f=f"{root}/posterior_samples")
        # if rd > 0:
        #     intersect_samples(network_storage=network_storage,
        #                       root_storage=root_storage, nreSettings=nreSettings,
        #                       obs=obs, rd=rd)
        newRoot = root + f"_rd_{rd + 1}"
        root = newRoot
        samples = nextSamples


if __name__ == '__main__':
    execute()
