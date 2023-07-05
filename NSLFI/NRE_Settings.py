class NRE_Settings:
    def __init__(self):
        """NRE initialisation.

        Settings for the (M)NRE algorithm

        """
        self.root = "swyft_polychord_NSNRE"
        self.wandb_project_name = "NSNRE"
        self.neural_network_file = "NRE_network.pt"
        self.activate_wandb = False
        self.logger_name = f"{self.root}.log"
        self.seed = 234
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
        self.max_epochs = 50
        self.NRE_num_retrain_rounds = 2
        self.activate_NSNRE_counting = False
        self.thinning_factor_of_dataset = 1
        self.ns_sampler = "Slice"
        self.ns_keep_chain = True
        self.ns_stopping_criterion = 1e-3
        self.sim_prior_lower = -1
        self.prior_width = 2
        self.anesthetic_sample_threshold = 0.999
        # polychord settings
        self.nderived = 0
        self.file_root = "samples"
        self.enhanced_run_file_root = "enhanced_run_samples"
        self.nlives_logL_coefficient = 0.05
