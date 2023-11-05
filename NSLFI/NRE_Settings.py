import numpy as np


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
        # simulator settings
        self.n_training_samples = 30_0
        self.n_weighted_samples = 30_000
        self.obsKey = "x"
        self.targetKey = "z"
        self.contourKey = "l"
        self.posteriorsKey = "post"
        self.num_features = 4
        self.num_features_dataset = 10
        self.num_mixture_components = 4
        self.sim_prior_lower = -10
        self.prior_width = 20
        # network settings
        self.device = "cpu"
        self.dropout = 0.3
        self.early_stopping_patience = 3
        self.max_epochs = 50
        self.NRE_num_retrain_rounds = 5
        self.NRE_start_from_round = 0
        self.learning_rate_init = 0.001
        self.learning_rate_decay = 0.1
        self.datamodule_fractions = [0.8, 0.1, 0.1]
        self.cyclic_rounds = True
        # polychord settings
        self.nlive_scan_run_per_feature = 1000
        self.nderived = 0
        self.file_root = "samples"
        # plotting settings
        self.only_plot_mode = False
        self.true_contours_available = True
        self.plot_triangle_plot = True
        self.plot_triangle_plot_ext = False
        self.plot_KL_divergence = True
        self.plot_quantile_plot = True
        self.percentiles_of_quantile_plot = np.arange(0, 1.05, 0.05)
        self.n_compressed_weighted_samples = 100
