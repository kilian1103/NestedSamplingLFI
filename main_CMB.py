import logging

import matplotlib.pyplot as plt
import pypolychord
from cmblike.cmb import CMB
from cmblike.noise import planck_noise
from mpi4py import MPI
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from NSLFI.NRE_Network_CMB import Network
from NSLFI.NRE_Post_Analysis import plot_analysis_of_NSNRE
from NSLFI.NRE_Simulator_CMB import Simulator
from NSLFI.PolySwyft import PolySwyft
from NSLFI.utils import *


def main():
    comm_gen = MPI.COMM_WORLD
    rank_gen = comm_gen.Get_rank()
    size_gen = comm_gen.Get_size()

    nreSettings = NRE_Settings()
    seed_everything(nreSettings.seed, workers=True)
    logging.basicConfig(filename=nreSettings.logger_name, level=logging.INFO,
                        filemode="w")
    logger = logging.getLogger()
    nreSettings.logger = logger
    logger.info('Started')

    # cosmopower params
    cp = True
    params = ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']  # cosmopower
    nreSettings.num_features = len(params)
    prior_mins = [0.005, 0.08, 0.01, 0.8, 2.6, 0.5]
    prior_maxs = [0.04, 0.21, 0.16, 1.2, 3.8, 0.9]
    cmbs = CMB(path_to_cp="../cosmopower", parameters=params, prior_maxs=prior_maxs, prior_mins=prior_mins)

    # prepare binning
    l_max = 2508  # read from planck unbinned data
    first_bin_width = 1
    second_bin_width = 30
    divider = 30
    first_bins = np.array([np.arange(2, divider, first_bin_width), np.arange(2, divider, first_bin_width)]).T  # 2 to 29
    second_bins = np.array([np.arange(divider, l_max - second_bin_width, second_bin_width),
                            np.arange(divider + second_bin_width, l_max, second_bin_width)]).T  # 30 to 2508
    last_bin = np.array([[second_bins[-1, 1], l_max]])  # remainder
    bins = np.concatenate([first_bins, second_bins, last_bin])
    bin_centers = np.concatenate([first_bins[:, 0], np.mean(bins[divider - 2:], axis=1)])

    # planck noise
    pnoise = planck_noise(bin_centers).calculate_noise()

    # binned planck data
    planck = np.loadtxt('data/planck_unbinned.txt', usecols=[1])
    planck = cmbs.rebin(planck, bins=bins)
    l = bin_centers.copy()
    nreSettings.num_features_dataset = len(l)

    sim = Simulator(nreSettings=nreSettings, cmbs=cmbs, bins=bins, bin_centers=bin_centers, p_noise=pnoise, cp=cp)
    obs = swyft.Sample(x=torch.as_tensor(planck)[None, :])

    # ['omegabh2', 'omegach2', 'tau', 'ns', 'As', 'h']
    theta_true = np.array([0.022, 0.12, 0.055, 0.965, 3.0, 0.67])
    sample_true = sim.sample(conditions={nreSettings.targetKey: theta_true})
    # obs = swyft.Sample(x=torch.as_tensor(sample_true[nreSettings.obsKey])[None, :])

    plt.plot(bin_centers, sample_true[nreSettings.obsKey], label="best fit sample")
    plt.plot(bin_centers, planck, label="planck")
    for i in range(3):
        sample = sim.sample()
        plt.plot(bin_centers, sample[nreSettings.obsKey])
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_{\ell}/2\pi$')
    plt.legend()
    plt.savefig("sim_planck_samples.pdf")
    plt.close()

    n_per_core = nreSettings.n_training_samples // size_gen
    seed_everything(nreSettings.seed + rank_gen, workers=True)
    if rank_gen == 0:
        n_per_core += nreSettings.n_training_samples % size_gen
    deadpoints = sim.sample(n_per_core, targets=[nreSettings.targetKey])[
        nreSettings.targetKey]
    comm_gen.Barrier()
    seed_everything(nreSettings.seed, workers=True)
    deadpoints = comm_gen.allgather(deadpoints)
    deadpoints = np.concatenate(deadpoints, axis=0)
    comm_gen.Barrier()

    network = Network(nreSettings=nreSettings, obs=obs, cmbs=cmbs)

    #### create callbacks function for pytorch lightning trainer
    def create_callbacks() -> list:
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.,
                                                patience=nreSettings.early_stopping_patience, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename='NRE_{epoch}_{val_loss:.2f}_{train_loss:.2f}', mode='min')
        return [early_stopping_callback, lr_monitor, checkpoint_callback]

    #### set up polychord settings
    polyset = pypolychord.PolyChordSettings(nreSettings.num_features, nDerived=nreSettings.nderived)
    polyset.file_root = "samples"
    polyset.base_dir = nreSettings.root
    polyset.seed = nreSettings.seed
    polyset.nfail = nreSettings.n_training_samples
    polySwyft = PolySwyft(nreSettings=nreSettings, sim=sim, obs=obs, deadpoints=deadpoints,
                          network=network, polyset=polyset, callbacks=create_callbacks)
    del deadpoints

    if not nreSettings.only_plot_mode:
        ### execute main cycle of NSNRE
        polySwyft.execute_NSNRE_cycle()

    root_storage, network_storage, samples_storage, dkl_storage = reload_data_for_plotting(nreSettings=nreSettings,
                                                                                           network=network,
                                                                                           polyset=polyset,
                                                                                           until_round=nreSettings.NRE_num_retrain_rounds)

    if rank_gen == 0:
        # plot analysis of NSNSRE
        plot_analysis_of_NSNRE(nreSettings=nreSettings, network_storage=network_storage,
                               samples_storage=samples_storage, dkl_storage=dkl_storage,
                               obs=obs, root=nreSettings.root)
    logger.info('Finished')


if __name__ == '__main__':
    main()
