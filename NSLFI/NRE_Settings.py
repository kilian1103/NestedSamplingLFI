class NRE_Settings:
    def __init__(self):
        """NRE initialisation.

        Settings for the (M)NRE algorithm

        """
        self.root = "swyft_torch_slice_fast"
        self.wandb_project_name = "NSNRE"
        self.logger_name = "myLFI.log"
        self.n_training_samples = 30_000
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
        self.NRE_num_retrain_rounds = 2
        self.ns_sampler = "Slice"
        self.ns_keep_chain = True
        self.ns_stopping_criterion = 1e-3
        self.ns_median_mode = True
        self.sim_prior_lower = -1
        self.sim_prior_upper = 2
