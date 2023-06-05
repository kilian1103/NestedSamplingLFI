import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import swyft
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from swyft import collate_output as reformat_samples
from torch import Tensor
from torch.distributions import Uniform

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator


def retrain_next_round(root: str, nextRoundPoints: Tensor, nreSettings: NRE_Settings,
                       sim: Simulator, prior: Dict[str, Uniform],
                       obs: swyft.Sample) -> Network:
    logger = logging.getLogger(nreSettings.logger_name)
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    logger.info(f"Simulating new {nreSettings.obsKey} using NS samples {nreSettings.targetKey} with Simulator!")
    samples = []
    for point in nextRoundPoints:
        cond = {nreSettings.targetKey: point}
        sample = sim.sample(conditions=cond, targets=[nreSettings.obsKey])
        samples.append(sample)
    samples = reformat_samples(samples)
    nextRoundSwyftSamples = swyft.Samples(samples)
    logger.info("Simulation done!")
    logger.info("Setting up network for training!")
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                            patience=nreSettings.early_stopping_patience, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=root,
                                          filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=root)
    trainer = swyft.SwyftTrainer(accelerator=nreSettings.device, devices=1, max_epochs=nreSettings.max_epochs,
                                 precision=64,
                                 enable_progress_bar=True,
                                 default_root_dir=nreSettings.root, logger=tb_logger,
                                 callbacks=[early_stopping_callback, lr_monitor,
                                            checkpoint_callback])
    dm = swyft.SwyftDataModule(nextRoundSwyftSamples, fractions=nreSettings.datamodule_fractions, num_workers=0,
                               batch_size=64)
    network = Network(nreSettings=nreSettings)
    # network = torch.compile(network)
    logger.info("Starting training!")
    trainer.fit(network, dm)
    logger.info("Training done!")
    if nreSettings.activate_wandb:
        wandb.finish()
    # get posterior samples
    logger.info("Sampling from the prior using simulator!")
    prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=[nreSettings.targetKey])
    logger.info("Inferring posterior samples using the trained network!")
    predictions = trainer.infer(network, obs, prior_samples)
    logger.info("Plotting posterior inference results!")
    plt.figure()
    swyft.corner(predictions, [f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)], bins=50,
                 smooth=1)
    plt.savefig(f"{root}/NRE_predictions.pdf")
    return network
