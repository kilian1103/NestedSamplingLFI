class NRE_Settings:
    def __init__(self, base_path):
        """NRE initialisation.

        Settings for the (M)NRE algorithm

        """
        self.trainmode = True
        self.device = "cpu"
        self.n_training_samples = 10_000
        self.n_weighted_samples = 10_000
        self.base_path = base_path
        self.observation_filename = f"{self.base_path}/observation.npy"
        self.obsKey = "x"
        self.targetKey = "z"
        self.dropout = 0.3
