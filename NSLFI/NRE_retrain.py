import logging
import os

import matplotlib.pyplot as plt
import swyft
import torch
import wandb
from swyft import collate_output as reformat_samples, Simulator
from torch import Tensor

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.utils import get_swyft_dataset_fractions


def retrain_next_round(root: str, training_data: Tensor, nreSettings: NRE_Settings,
                       sim: Simulator,
                       obs: swyft.Sample, network: swyft.SwyftModule,
                       dm: swyft.SwyftDataModule, trainer: swyft.SwyftTrainer, rd: int) -> swyft.SwyftModule:
    logger = logging.getLogger(nreSettings.logger_name)
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    logger.info(f"Simulating new {nreSettings.obsKey} using NS samples {nreSettings.targetKey} with Simulator!")
    samples = []
    for point in training_data:
        cond = {nreSettings.targetKey: point.float()}
        if nreSettings.use_noise_resampling and rd > 0:
            resampler = sim.get_resampler(targets=[nreSettings.obsKey])
            for _ in range(nreSettings.n_noise_resampling_samples):
                cond[nreSettings.obsKey] = None
                sample = resampler(cond)
                samples.append(sample)
        else:
            sample = sim.sample(conditions=cond, targets=[nreSettings.obsKey])
            samples.append(sample)
    samples = reformat_samples(samples)
    training_data_swyft = swyft.Samples(samples)
    logger.info("Simulation done!")
    logger.info("Setting up network for training!")
    # network = torch.compile(network)
    logger.info("Starting training!")
    dm.data = training_data_swyft
    dm.lengths = get_swyft_dataset_fractions(nreSettings.datamodule_fractions, len(training_data_swyft))
    trainer.fit(network, dm)
    logger.info("Training done!")
    if nreSettings.activate_wandb:
        wandb.finish()
    # get posterior samples
    logger.info("Sampling from the prior using simulator!")
    prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=[nreSettings.targetKey])
    logger.info("Inferring posterior samples using the trained network!")
    with torch.no_grad():
        predictions = trainer.infer(network, obs, prior_samples)
    logger.info("Plotting posterior inference results!")
    plt.figure()
    swyft.corner(predictions, [f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)], bins=50,
                 smooth=1)
    plt.savefig(f"{root}/NRE_predictions.pdf")
    torch.save(network.state_dict(), f"{root}/{nreSettings.neural_network_file}")
    return network
