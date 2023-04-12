import os
from logging import Logger
from typing import Dict

import matplotlib.pyplot as plt
import swyft
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from swyft import collate_output
from torch import Tensor
from torch.distributions import Uniform

import NSLFI.NestedSampler
from NSLFI.NRE_NS_Wrapper import NRE
from NSLFI.NRE_Network import Network
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator


def retrain_next_round(root: str, nextRoundPoints: Tensor, nreSettings: NRE_Settings, sim: Simulator,
                       prior: Dict[str, Uniform],
                       logger: Logger, obs: swyft.Sample) -> Network:
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    out = []
    logger.info(f"Simulating new {nreSettings.obsKey} using NS samples {nreSettings.targetKey} with Simulator!")
    for z in nextRoundPoints.numpy().squeeze():
        trace = dict()
        trace[nreSettings.targetKey] = z
        sim.graph[nreSettings.obsKey].evaluate(trace)
        sim.graph["l"].evaluate(trace)
        result = sim.transform_samples(trace)
        out.append(result)
    out = collate_output(out)
    nextRoundSamples = swyft.Samples(out)
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
    dm = swyft.SwyftDataModule(nextRoundSamples, fractions=nreSettings.datamodule_fractions, num_workers=0,
                               batch_size=64)
    network = Network(nreSettings=nreSettings)
    # network = torch.compile(network)
    logger.info("Starting training!")
    trainer.fit(network, dm)
    logger.info("Training done!")
    # get posterior samples
    logger.info("Sampling from the prior using simulator!")
    prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=[nreSettings.targetKey])
    logger.info("Inferring posterior samples using the trained network!")
    predictions = trainer.infer(network, obs, prior_samples)
    logger.info("Plotting posterior inference results!")
    plt.figure()
    swyft.corner(predictions, ["z[0]", "z[1]"], bins=50, smooth=1)
    plt.savefig(f"{root}/NRE_predictions.pdf")
    plt.show()
    # wrap NRE object
    trained_NRE = NRE(network=network, obs=obs, livepoints=nextRoundPoints)
    logger.info("Using Nested Sampling and trained NRE to generate new samples for the next round!")
    with torch.no_grad():
        output = NSLFI.NestedSampler.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                                     livepoints=trained_NRE.livepoints, prior=prior, nsim=100,
                                                     stop_criterion=nreSettings.ns_stopping_criterion,
                                                     round_mode=nreSettings.ns_round_mode,
                                                     num_rounds=nreSettings.ns_num_rounds,
                                                     root=root,
                                                     samplertype=nreSettings.ns_sampler,
                                                     nsamples=nreSettings.n_training_samples,
                                                     keep_chain=nreSettings.ns_keep_chain)
    return network
