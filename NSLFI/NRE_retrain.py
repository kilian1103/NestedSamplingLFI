import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import swyft
import torch
import wandb
from swyft import collate_output as reformat_samples, Simulator
from torch import Tensor

from NSLFI.NRE_Settings import NRE_Settings


def retrain_next_round(root: str, training_data: Tensor, nreSettings: NRE_Settings,
                       sim: Simulator,
                       obs: swyft.Sample, untrained_network: swyft.SwyftModule,
                       trainer: swyft.SwyftTrainer, dm: swyft.SwyftDataModule) -> swyft.SwyftModule:
    logger = logging.getLogger(nreSettings.logger_name)
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    logger.info(f"Simulating new {nreSettings.obsKey} using NS samples {nreSettings.targetKey} with Simulator!")
    samples = []
    for point in training_data:
        cond = {nreSettings.targetKey: point.float()}
        sample = sim.sample(conditions=cond, targets=[nreSettings.obsKey])
        samples.append(sample)
    samples = reformat_samples(samples)
    training_data_swyft = swyft.Samples(samples)
    logger.info("Simulation done!")
    logger.info("Setting up network for training!")
    network = untrained_network.get_new_network()
    # network = torch.compile(network)
    logger.info("Starting training!")
    dm.data = training_data_swyft
    trainer.fit(network, dm)
    logger.info("Training done!")
    if nreSettings.activate_wandb:
        wandb.finish()
    # get posterior samples
    trainer.save_checkpoint(filepath=f"{root}")
    logger.info("Sampling from the prior using simulator!")
    # TODO prior of simulator is not full prior
    # prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=[nreSettings.targetKey])
    prior_samples = np.random.uniform(nreSettings.sim_prior_lower,
                                      nreSettings.sim_prior_lower + nreSettings.prior_width,
                                      size=(nreSettings.n_weighted_samples, nreSettings.num_features))
    prior_samples = swyft.Samples({nreSettings.targetKey: prior_samples})
    logger.info("Inferring posterior samples using the trained network!")
    predictions = trainer.infer(network, obs, prior_samples)
    logger.info("Plotting posterior inference results!")
    plt.figure()
    swyft.corner(predictions, [f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)], bins=50,
                 smooth=1)
    plt.savefig(f"{root}/NRE_predictions.pdf")
    torch.save(network.state_dict(), f"{root}/{nreSettings.neural_network_file}")
    return network
