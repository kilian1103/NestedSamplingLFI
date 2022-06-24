import numpy as np


class NRE_Settings:
    def __init__(self):
        """NRE initialisation.

        Settings for the (M)NRE algorithm

        """
        self.mode = "load"
        self.MNREmode = False
        self.simulatedObservations = False
        self.device = "cpu"
        self.n_training_samples = 10_000
        self.n_weighted_samples = 10_000
        self.theta_0 = np.array([5, 25, 10])
        self.paramNames = [r"$\sigma$", r"$f_0$", r"$A$"]
        self.n_parameters = len(self.paramNames)
        self.prior_filename = "swyft_data/toyproblem.prior.pt"
        self.dataset_filename = "swyft_data/toyproblem.dataset.pt"
        self.dataset_filename_NSenhanced = "swyft_data/toyproblem.dataset_NSenhanced.pt"
        self.mre_1d_filename = "swyft_data/toyproblem.mre_1d.pt"
        self.mre_2d_filename = "swyft_data/toyproblem.mre_2d.pt"
        self.mre_3d_filename = "swyft_data/toyproblem.mre_3d.pt"
        self.mre_3d_filename_NSenhanced = "swyft_data/toyproblem.mre_3d_NSenhanced.pt"
        self.store_filename = "swyft_data/SavedStore"
        self.store_filename_NSenhanced = "swyft_data/SavedStore_NSenhanced"
        self.observation_filename = "swyft_data/observation.npy"
        self.observation_key = "x"
