class NRE_Settiings:
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
        self.paramNames = [r"$\sigma$", r"$f_0$", r"$A$"]
        self.ndim = len(self.paramNames)
        self.prior_filename = "swyft_data/toyproblem.prior.pt"
        self.dataset_filename = "swyft_data/toyproblem.dataset.pt"
        self.mre_1d_filename = "swyft_data/toyproblem.mre_1d.pt"
        self.mre_2d_filename = "swyft_data/toyproblem.mre_2d.pt"
        self.mre_3d_filename = "swyft_data/toyproblem.mre_3d.pt"
        self.store_filename = "swyft_data/SavedStore"
        self.observation_filename = "swyft_data/observation.npy"
        self.observation_key = "x"
