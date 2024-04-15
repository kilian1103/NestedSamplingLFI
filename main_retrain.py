import copy
import os

import anesthetic
import numpy as np
import pypolychord
import sklearn
import swyft
import torch
from mpi4py import MPI
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from swyft import collate_output as reformat_samples

from NSLFI.NRE_Network import Network
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NRE_Simulator_MixGauss import Simulator


def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()
    root = "swyft_example_099_contour_GMM_100d_1klive_samples_10Noise"
    roots = [f"{root}_round_{x}" for x in range(0, 5, 1)]

    it = 4  # NRE to be retrained

    nreSettings = NRE_Settings()
    np.random.seed(nreSettings.seed)
    torch.manual_seed(nreSettings.seed)
    #### instantiate swyft simulator
    n = nreSettings.num_features
    d = nreSettings.num_features_dataset
    a = nreSettings.num_mixture_components

    mu_data = torch.randn(size=(a, d)) * 3  # random mean vec of data
    M = torch.randn(size=(a, d, n))  # random transform matrix of param to data space vec
    C = torch.eye(d)  # cov matrix of dataset
    # mu_theta = torch.randn(size=(1, n))  # random mean vec of parameter
    mu_theta = torch.randn(size=(a, n)) * 3  #
    Sigma = torch.eye(n)  # cov matrix of parameter prior
    sim = Simulator(nreSettings=nreSettings, mu_theta=mu_theta, M=M, mu_data=mu_data, Sigma=Sigma, C=C)
    nreSettings.model = sim.model  # lsbi model

    # generate training dat and obs
    obs = swyft.Sample(x=torch.tensor(sim.model.evidence().rvs()[None, :]))

    network = Network(nreSettings=nreSettings, obs=obs)
    dm = swyft.SwyftDataModule(data=[0], **nreSettings.dm_kwargs)
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                            patience=nreSettings.early_stopping_patience, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')

    trainer = swyft.SwyftTrainer(accelerator=nreSettings.device, devices=1, max_epochs=nreSettings.max_epochs,
                                 precision=64,
                                 enable_progress_bar=True,
                                 default_root_dir=nreSettings.root,
                                 callbacks=[early_stopping_callback, lr_monitor, checkpoint_callback])

    polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
    polyset.seed = nreSettings.seed
    polyset.nfail = nreSettings.nlives_per_dim_constant * nreSettings.n_prior_sampling
    polyset.nprior = nreSettings.n_prior_sampling
    polyset.file_root = "samples"

    training_root = roots[it - 1]
    samples = []

    if nreSettings.use_livepoint_increasing:
        try:
            training_data = torch.load(
                f=f"{training_root}/{nreSettings.increased_livepoints_fileroot}/"
                  f"{nreSettings.joint_training_data_fileroot}")
        except FileNotFoundError:
            deadpoints = anesthetic.read_chains(
                root=f"{training_root}/{nreSettings.increased_livepoints_fileroot}/{polyset.file_root}")
            deadpoints = deadpoints.iloc[:, :nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            for point in deadpoints:
                cond = {nreSettings.targetKey: point.float()}
                if nreSettings.use_noise_resampling:
                    resampler = sim.get_resampler(targets=[nreSettings.obsKey])
                    for _ in range(nreSettings.n_noise_resampling_samples):
                        cond[nreSettings.obsKey] = None
                        sample = resampler(cond)
                        samples.append(sample)

            samples = sklearn.utils.shuffle(samples, random_state=nreSettings.seed)
            training_data = reformat_samples(samples)
            torch.save(
                f=f"{training_root}/{nreSettings.increased_livepoints_fileroot}/"
                  f"{nreSettings.joint_training_data_fileroot}",
                obj=training_data)


    else:
        try:
            training_data = torch.load(f=f"{training_root}/{nreSettings.joint_training_data_fileroot}")
        except FileNotFoundError:
            deadpoints = anesthetic.read_chains(
                root=f"{training_root}/{polyset.file_root}")
            deadpoints = deadpoints.iloc[:, :nreSettings.num_features]
            deadpoints = torch.as_tensor(deadpoints.to_numpy())
            for point in deadpoints:
                cond = {nreSettings.targetKey: point.float()}
                if nreSettings.use_noise_resampling:
                    resampler = sim.get_resampler(targets=[nreSettings.obsKey])
                    for _ in range(nreSettings.n_noise_resampling_samples):
                        cond[nreSettings.obsKey] = None
                        sample = resampler(cond)
                        samples.append(sample)

            samples = sklearn.utils.shuffle(samples, random_state=nreSettings.seed)
            training_data = reformat_samples(samples)
            torch.save(
                f=f"{training_root}/{nreSettings.joint_training_data_fileroot}",
                obj=training_data)

    training_data_swyft = swyft.Samples(training_data)
    dm = swyft.SwyftDataModule(data=training_data_swyft, **nreSettings.dm_kwargs)

    new_trainer = copy.deepcopy(trainer)
    new_network = network.get_new_network()
    nreSettings.neural_network_file = f"NRE_network_{rank_gen}_retrained.pt"
    new_trainer.fit(new_network, dm)
    root = roots[it]
    try:
        os.makedirs(root)
    except OSError:
        print("root folder already exists!")

    torch.save(new_network.state_dict(), f"{root}/{nreSettings.neural_network_file}")
    comm_gen.Barrier()

    for i in range(size_gen):
        nreSettings.neural_network_file = f"NRE_network_{i}_retrained.pt"
        new_network = network.get_new_network()
        new_network.double()  # change to float64 precision of network
        new_network.load_state_dict(torch.load(f"{root}/{nreSettings.neural_network_file}"))
        new_network.eval()

        ### start polychord section ###
        ### Run PolyChord ###
        polyset.base_dir = root
        polyset.file_root = f"samples_{i}_retrain"
        polyset.nlive = nreSettings.nlives_per_round[0]
        comm_gen.barrier()
        pypolychord.run_polychord(loglikelihood=new_network.logLikelihood,
                                  nDims=nreSettings.num_features,
                                  nDerived=nreSettings.nderived, settings=polyset,
                                  prior=new_network.prior, dumper=new_network.dumper)
        comm_gen.Barrier()


if __name__ == '__main__':
    main()
