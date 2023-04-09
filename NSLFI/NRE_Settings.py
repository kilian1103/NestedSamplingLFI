class NRE_Settings:
    def __init__(self, base_path):
        """NRE initialisation.

        Settings for the (M)NRE algorithm

        """
        self.base_path = base_path
        self.n_training_samples = 10_000
        self.n_weighted_samples = 10_000
        self.datamodule_fractions = [0.8, 0.1, 0.1]
        self.obsKey = "x"
        self.targetKey = "z"
        self.num_features = 2
        self.trainmode = True
        self.device = "cpu"
        self.dropout = 0.3
        self.early_stopping_patience = 3
        self.max_epochs = 20
        self.NRE_num_retrain_rounds = 3
        self.wandb_project_name = "NSNRE_REEFACTOR"
        self.observation_filename = f"{self.base_path}/observation.npy"
        self.ns_sampler = "Slice"
        self.ns_round_mode = False
        self.ns_num_rounds = 3
        self.ns_keep_chain = True
        self.ns_stopping_criterion = 1e-3
