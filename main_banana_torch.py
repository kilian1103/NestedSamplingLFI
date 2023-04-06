import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import swyft
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from swyft import collate_output

import NSLFI.NestedSamplerTorch
import wandb
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.Swyft_NRE_Wrapper import NRE


# from NSLFI.Swyft_NRE_Wrapper import NRE
def execute():
    np.random.seed(234)
    torch.manual_seed(234)
    logging.basicConfig(filename="myLFI.log", level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    logger.info('Started')
    dropout = 0.3
    root = "swyft_torch_slice_fast"
    try:
        os.makedirs(root)
    except OSError:
        logger.info("root folder already exists!")

    nreSettings = NRE_Settings(base_path=root)
    nreSettings.n_training_samples = 30_000
    nreSettings.n_weighted_samples = 10_000
    nreSettings.trainmode = True
    wandb_project_name = "NSNRE"
    # NS rounds, 0 is default NS run
    rounds = 1
    # Retrain rounds
    retrain_rounds = 2
    keep_chain = True
    samplerType = "Slice"
    # define forward model dimensions
    bimodal = False
    nParam = 2
    # true parameters of simulator
    obs = swyft.Sample(x=np.array(nParam * [0]))
    # uniform prior for theta_i
    lower = -1
    upper = 2
    theta_prior = torch.distributions.uniform.Uniform(low=lower, high=lower + upper)
    # wrap prior for NS sampling procedure
    prior = {f"theta_{i}": theta_prior for i in range(nParam)}

    class Simulator(swyft.Simulator):
        def __init__(self, bounds_z=None, bimodal=bimodal):
            super().__init__()
            self.z_sampler = swyft.RectBoundSampler(
                [stats.uniform(lower, upper),
                 stats.uniform(lower, upper),
                 ],
                bounds=bounds_z
            )
            self.bimodal = bimodal

        def f(self, z):
            if self.bimodal:
                if z[0] < 0:
                    z = np.array([z[0] + 0.5, z[1] - 0.5])
                else:
                    z = np.array([z[0] - 0.5, -z[1] - 0.5])
            z = 10 * np.array([z[0], 10 * z[1] + 100 * z[0] ** 2])
            return z

        def build(self, graph):
            z = graph.node("z", self.z_sampler)
            x = graph.node("x", lambda z: self.f(z) + np.random.randn(2), z)
            l = graph.node("l", lambda z: -stats.norm.logpdf(self.f(z)).sum(),
                           z)  # return -ln p(x=0|z) for cross-checks

    sim = Simulator(bounds_z=None, bimodal=bimodal)
    samples = sim.sample(nreSettings.n_training_samples)
    dm = swyft.SwyftDataModule(samples, fractions=[0.8, 0.1, 0.1], num_workers=0, batch_size=64)
    plt.tricontour(samples['z'][:, 0], samples['z'][:, 1], samples['l'] - samples['l'].min(), levels=[0, 1, 4])
    if bimodal:
        plt.ylim(-0.55, 0.55)
    else:
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.8, 0.1)

    class Network(swyft.SwyftModule):
        def __init__(self):
            super().__init__()
            #  self.logratios1 = swyft.LogRatioEstimator_1dim(num_features=2, num_params=2, varnames='z',
            #  dropout=0.2, hidden_features=128)
            self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features=2, marginals=((0, 1),), varnames='z',
                                                           dropout=dropout, hidden_features=128, Lmax=8)

        def forward(self, A, B):
            return self.logratios2(A['x'], B['z'])

    network = Network()
    # network = torch.compile(network)
    wandb.init(
        # set the wandb project where this run will be logged
        project=wandb_project_name, name="round_0", sync_tensorboard=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0., patience=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=root,
                                          filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=root)

    trainer = swyft.SwyftTrainer(accelerator='cpu', devices=1, max_epochs=20, precision=64, enable_progress_bar=True,
                                 default_root_dir=nreSettings.base_path, logger=tb_logger,
                                 callbacks=[early_stopping_callback, lr_monitor,
                                            checkpoint_callback])
    # train MRE
    if nreSettings.trainmode is True:
        trainer.fit(network, dm)

    # load NRE from file
    else:
        checkpoint_path = os.path.join(
            f"swyft_torch_test_slice/lightning_logs/version_16795008/checkpoints/epoch=6-step=2625.ckpt")
        network = network.load_from_checkpoint(checkpoint_path)
    wandb.finish()
    # get posterior samples
    prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=['z'])
    predictions = trainer.infer(network, obs, prior_samples)
    plt.figure()
    swyft.corner(predictions, ["z[0]", "z[1]"], bins=50, smooth=1)
    plt.savefig(f"{root}/NRE_predictions.pdf")
    plt.show()

    # wrap NRE object
    trained_NRE = NRE(network=network, prior=prior, nreSettings=nreSettings, obs=obs,
                      livepoints=torch.tensor(samples["z"]))
    with torch.no_grad():
        output = NSLFI.NestedSamplerTorch.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                                          livepoints=trained_NRE.livepoints, prior=prior, nsim=100,
                                                          stop_criterion=1e-3,
                                                          samplertype=samplerType, rounds=rounds, root=root,
                                                          nsamples=nreSettings.n_training_samples,
                                                          keep_chain=keep_chain)

    def retrain_next_round(root: str, nextRoundPoints: torch.tensor):
        try:
            os.makedirs(root)
        except OSError:
            logger.info("root folder already exists!")
        out = []
        for z in nextRoundPoints.numpy().squeeze():
            trace = dict()
            trace["z"] = z
            sim.graph["x"].evaluate(trace)
            sim.graph["l"].evaluate(trace)
            result = sim.transform_samples(trace)
            out.append(result)
        out = collate_output(out)
        nextRoundSamples = swyft.Samples(out)
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0., patience=3, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=root,
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=root)
        trainer = swyft.SwyftTrainer(accelerator='cpu', devices=1, max_epochs=20, precision=64,
                                     enable_progress_bar=True,
                                     default_root_dir=nreSettings.base_path, logger=tb_logger,
                                     callbacks=[early_stopping_callback, lr_monitor,
                                                checkpoint_callback])
        dm = swyft.SwyftDataModule(nextRoundSamples, fractions=[0.8, 0.1, 0.1], num_workers=0, batch_size=64)
        network = Network()
        # network = torch.compile(network)
        trainer.fit(network, dm)
        # get posterior samples
        prior_samples = sim.sample(nreSettings.n_weighted_samples, targets=['z'])
        predictions = trainer.infer(network, obs, prior_samples)
        plt.figure()
        swyft.corner(predictions, ["z[0]", "z[1]"], bins=50, smooth=1)
        plt.savefig(f"{root}/NRE_predictions.pdf")
        plt.show()
        # wrap NRE object
        trained_NRE = NRE(network=network, prior=prior, nreSettings=nreSettings, obs=obs,
                          livepoints=torch.tensor(nextRoundSamples["z"]))
        with torch.no_grad():
            output = NSLFI.NestedSamplerTorch.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                                              livepoints=trained_NRE.livepoints, prior=prior, nsim=100,
                                                              stop_criterion=1e-3, rounds=1,
                                                              root=root,
                                                              samplertype=samplerType,
                                                              nsamples=nreSettings.n_training_samples,
                                                              keep_chain=keep_chain)

    for rd in range(1, retrain_rounds + 1):
        logger.info("retraining round: " + str(rd))
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project_name, name=f"round_{rd}", sync_tensorboard=True)
        nextRoundPoints = torch.load(f=f"{root}/posterior_samples_rounds_0")
        newRoot = root + f"_rd_{rd}"
        retrain_next_round(newRoot, nextRoundPoints)
        root = newRoot
        wandb.finish()


if __name__ == '__main__':
    execute()
