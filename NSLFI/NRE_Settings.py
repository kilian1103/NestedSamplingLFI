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
        self.num_features = 5
        self.num_features_dataset = 15
        self.num_mixture_components = 4
        # network settings
        self.device = "cpu"
        self.dropout = 0.3
        self.early_stopping_patience = 3
        self.max_epochs = 50
        self.learning_rate_init = 0.001
        self.learning_rate_decay = 0.1
        self.datamodule_fractions = [0.8, 0.1, 0.1]
        # polychord settings
        self.nderived = 0
        self.flow = None
        self.model = None
        # NSNRE settings
        self.cyclic_rounds = True
        self.NRE_num_retrain_rounds = 5
        self.NRE_start_from_round = 0
        self.termination_abs_dkl = 0.2
        self.n_DKL_estimates = 100
        # plotting settings
        self.only_plot_mode = False
        self.true_contours_available = True
        self.plot_triangle_plot = True
        self.plot_triangle_plot_ext = False
        self.plot_KL_divergence = True
        self.plot_quantile_plot = True
        self.percentiles_of_quantile_plot = np.arange(0, 1.05, 0.05)
        self.n_compressed_weighted_samples = 100
