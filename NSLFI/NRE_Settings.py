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
        self.theta_0 = np.array([0.5, 0.5])
        self.paramNames = [r"$\mu_1$", r"$\mu_2$"]
        self.n_parameters = len(self.paramNames)
        self.base_path = "swyft_data"
        self.prior_filename = f"{self.base_path}/toyproblem.prior.pt"
        self.dataset_filename = f"{self.base_path}/toyproblem.dataset.pt"
        self.dataset_filename_NSenhanced = f"{self.base_path}/toyproblem.dataset_NSenhanced.pt"
        self.mre_1d_filename = f"{self.base_path}/toyproblem.mre_1d.pt"
        self.mre_2d_filename = f"{self.base_path}/toyproblem.mre_2d.pt"
        self.mre_3d_filename = f"{self.base_path}/toyproblem.mre_3d.pt"
        self.mre_3d_filename_NSenhanced = f"{self.base_path}/toyproblem.mre_3d_NSenhanced.pt"
        self.mre_2d_filename_NSenhanced = f"{self.base_path}/toyproblem.mre_2d_NSenhanced.pt"
        self.store_filename = f"{self.base_path}/SavedStore"
        self.store_filename_NSenhanced = f"{self.base_path}/SavedStore_NSenhanced"
        self.observation_filename = f"{self.base_path}/observation.npy"
        self.observation_key = "x"
