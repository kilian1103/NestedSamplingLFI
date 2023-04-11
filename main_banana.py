import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import swyft
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from swyft import collate_output
from torch import Tensor

import NSLFI.NestedSampler
import wandb
from NSLFI.NRE_NS_Wrapper import NRE
from NSLFI.NRE_Network import Network
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator import Simulator


# from NSLFI.Swyft_NRE_Wrapper import NRE
def execute():
    np.random.seed(234)
    torch.manual_seed(234)
    logging.basicConfig(filename="myLFI.log", level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    nreSettings = NRE_Settings()
    root = nreSettings.root
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")
    nreSettings.n_training_samples = 30_000
    nreSettings.n_weighted_samples = 10_000
    nreSettings.trainmode = True
    obskey = nreSettings.obsKey
    targetkey = nreSettings.targetKey
    dropout = nreSettings.dropout
    wandb_project_name = nreSettings.wandb_project_name
    retrain_rounds = nreSettings.NRE_num_retrain_rounds
    network_storage = dict()

    # NS settings, 0 is default NS run
    round_mode = nreSettings.ns_round_mode
    keep_chain = nreSettings.ns_keep_chain
    samplerType = nreSettings.ns_sampler

    # define forward model settings
    nParam = nreSettings.num_features
    bimodal = False
    # observatioon for simulator
    obs = swyft.Sample(x=np.array(nParam * [0]))
    # uniform prior for theta_i
    lower = nreSettings.sim_prior_lower
    upper = nreSettings.sim_prior_upper
    theta_prior = torch.distributions.uniform.Uniform(low=lower, high=lower + upper)
    # wrap prior for NS sampling procedure
    prior = {f"theta_{i}": theta_prior for i in range(nParam)}

    sim = Simulator(bounds_z=None, bimodal=bimodal, nreSettings=nreSettings)
    samples = torch.as_tensor(sim.sample(nreSettings.n_training_samples, targets=[targetkey])[targetkey])

    def retrain_next_round(root: str, nextRoundPoints: Tensor) -> Network:
        try:
            os.makedirs(root)
        except OSError:
            logger.info("root folder already exists!")
        out = []
        for z in nextRoundPoints.numpy().squeeze():
            trace = dict()
            trace[targetkey] = z
            sim.graph[obskey].evaluate(trace)
            sim.graph["l"].evaluate(trace)
            result = sim.transform_samples(trace)
            out.append(result)
        out = collate_output(out)
        nextRoundSamples = swyft.Samples(out)
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
        trainer.fit(network, dm)
        # get posterior samples
        prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=[targetkey])
        predictions = trainer.infer(network, obs, prior_samples)
        plt.figure()
        swyft.corner(predictions, ["z[0]", "z[1]"], bins=50, smooth=1)
        plt.savefig(f"{root}/NRE_predictions.pdf")
        plt.show()
        # wrap NRE object
        trained_NRE = NRE(network=network, nreSettings=nreSettings, obs=obs, livepoints=nextRoundPoints)
        with torch.no_grad():
            output = NSLFI.NestedSampler.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                                         livepoints=trained_NRE.livepoints, prior=prior, nsim=100,
                                                         stop_criterion=nreSettings.ns_stopping_criterion,
                                                         round_mode=round_mode,
                                                         num_rounds=nreSettings.ns_num_rounds,
                                                         root=root,
                                                         samplertype=samplerType,
                                                         nsamples=nreSettings.n_training_samples,
                                                         keep_chain=keep_chain)
        return network

    for rd in range(0, retrain_rounds + 1):
        logger.info("retraining round: " + str(rd))
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
        network = retrain_next_round(root=root, nextRoundPoints=samples)
        network_storage[f"round_{rd}"] = network
        wandb.finish()
        nextSamples = torch.load(f=f"{root}/posterior_samples_rounds_0")
        newRoot = root + f"_rd_{rd + 1}"
        root = newRoot
        samples = nextSamples


if __name__ == '__main__':
    execute()
