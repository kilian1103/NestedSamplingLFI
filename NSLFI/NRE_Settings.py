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
        self.num_features = 2
        self.dropout = 0.3
        self.early_stopping_patience = 3
        self.max_epochs = 20
        self.NRE_num_retrain_rounds = 3
