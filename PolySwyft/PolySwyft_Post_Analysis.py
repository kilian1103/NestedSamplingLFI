import os
from typing import Dict, Tuple, Callable

import anesthetic
import matplotlib.pyplot as plt
import numpy as np
import swyft
import torch
from anesthetic import make_2d_axes, make_1d_axes

from PolySwyft.PolySwyft_Settings import PolySwyft_Settings
from PolySwyft.utils import compute_KL_compression, compute_KL_divergence_truth



def plot_analysis_of_NSNRE(root: str, network_storage: Dict[int, swyft.SwyftModule],
                           samples_storage: Dict[int, anesthetic.Samples], dkl_storage: Dict[int, Tuple[float, float]],
                           polyswyftSettings: PolySwyft_Settings,
                           obs: swyft.Sample, true_posterior: anesthetic.Samples = None, deadpoints_processing: Callable = None):
    """
    Plot the analysis of the NSNRE.
    :param root: A string of the root directory to save the plots
    :param root_storage: A dictionary of roots for each round
    :param network_storage: A dictionary of networks for each round
    :param samples_storage: A dictionary of samples for each round
    :param dkl_storage: A dictionary of KL divergences for each round
    :param polyswyftSettings: A PolySwyft_Settings object
    :param obs: A Swyft sample of the observed data
    :param true_posterior: An anesthetic samples object of the true posterior if available
    :return:
    """
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', family='serif')

    # set up labels for plotting
    params_idx = [i for i in range(0, polyswyftSettings.num_features)]
    params_labels = {i: rf"${polyswyftSettings.targetKey}_{i}$" for i in range(polyswyftSettings.num_features)}

    dkl_storage_true = {}

    # triangle plot
    if polyswyftSettings.plot_triangle_plot:
        kinds = {'lower': 'kde_2d', 'diagonal': 'kde_1d', 'upper': "scatter_2d"}
        fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=True,
                                 ticks="outer")
        first_round_samples = samples_storage[0]
        # load prior from last round
        prior = first_round_samples.prior()
        prior.plot_2d(axes=axes, alpha=0.4, label="prior", kinds=kinds)
        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            nested.plot_2d(axes=axes, alpha=0.4, label=fr"$p(\theta|D)_{rd}$",
                           kinds=kinds)
        if true_posterior is not None:
            true_posterior.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                   kinds=kinds)

        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=polyswyftSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_full.pdf")

        ### zoomed in triangle plot
    if polyswyftSettings.plot_triangle_plot_zoomed:
        kinds = {'lower': 'kde_2d', 'diagonal': 'kde_1d', 'upper': "scatter_2d"}
        fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=True,
                                 ticks="outer")
        for rd in range(polyswyftSettings.triangle_zoom_start,
                        polyswyftSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            nested.plot_2d(axes=axes, alpha=0.4, label=fr"$p(\theta|D)_{rd}$",
                           kinds=kinds)
        if true_posterior is not None:
            true_posterior.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                   kinds=kinds)
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=polyswyftSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_zoom.pdf")
        plt.close()

    # KL divergence plot
    if polyswyftSettings.plot_KL_divergence:
        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            if true_posterior is not None:
                previous_network = network_storage[rd]
                KDL_true = compute_KL_divergence_truth(polyswyftSettings=polyswyftSettings,
                                                       previous_network=previous_network.eval(),
                                                       true_posterior=true_posterior.copy(), obs=obs,
                                                       previous_samples=samples_storage[rd])
                dkl_storage_true[rd] = KDL_true
        plt.figure(figsize=(3.5, 3.5))

        plt.errorbar(x=[i for i in range(1, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     y=[dkl_storage[i][0] for i in range(1, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     yerr=[dkl_storage[i][1] for i in range(1, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     label=r"$\mathrm{KL} (\mathcal{P}_i||\mathcal{P}_{i-1})$")
        if true_posterior is not None:
            plt.errorbar(x=[i for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                         y=[dkl_storage_true[i][0] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                         yerr=[dkl_storage_true[i][1] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                         label=r"$\mathrm{KL}(\mathcal{P}_{\mathrm{True}}||\mathcal{P}_i)$")
        dkl_compression_storage = {}
        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            DKL = compute_KL_compression(samples_storage[rd], polyswyftSettings)
            dkl_compression_storage[rd] = DKL
        plt.errorbar(x=[i for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     y=[dkl_compression_storage[i][0] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     yerr=[dkl_compression_storage[i][1] for i in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1)],
                     label=r"$\mathrm{KL}(\mathcal{P}_i||\pi)$")
        plt.legend()
        plt.xlabel("retrain round")
        plt.ylabel("KL divergence")
        plt.savefig(f"{root}/kl_divergence.pdf", dpi=300, bbox_inches='tight')
        plt.close()


    if polyswyftSettings.plot_logR_histogram:
        path = "logR_histogram"
        try:
            os.makedirs(f"{root}/{path}")
        except OSError:
            print(f"{path} folder already exists!")

        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            samples = samples_storage[rd]
            logRs = samples["logL"] - samples.logZ()
            plt.hist(samples["logL"], label=r"$\log r_{\mathrm{uncorr}}$", alpha=0.5, bins=50)
            plt.hist(logRs, label=r"$\log r_{\mathrm{corr}}$", alpha=0.5, bins=50)
            plt.title(r"$\log r$ histogram")
            plt.xlabel(r"$\log r$")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(f"{root}/{path}/logR_histogram_unweighted_{rd}.pdf")
            plt.close()

    if polyswyftSettings.plot_logR_pdf:
        path = "logR_pdf"
        try:
            os.makedirs(f"{root}/{path}")
        except OSError:
            print(f"{path}-folder already exists!")

        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            samples = samples_storage[rd]
            samples["logR"] = samples["logL"] - samples.logZ()
            figs, axes = make_1d_axes("logR", figsize=(3.5, 3.5))
            samples.plot_1d(axes=axes, label=f"round {rd}")
            plt.xlabel(r"$\log r$")
            plt.ylabel(r"$p(\log r)$")
            plt.legend()
            plt.savefig(f"{root}/{path}/logR_pdf_{rd}.pdf", dpi=300, bbox_inches='tight')
            plt.close()



    if polyswyftSettings.save_joint_training_data and polyswyftSettings.plot_training_data:
        path = "training_data"
        try:
            os.makedirs(f"{root}/{path}")
        except OSError:
            print(f"{path}-folder already exists!")

        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            joint = torch.load(f"{root}/{polyswyftSettings.child_root}_{rd}/{polyswyftSettings.joint_training_data_fileroot}")
            plt.figure()
            plt.scatter(joint[polyswyftSettings.targetKey][:, 0], joint[polyswyftSettings.targetKey][:, 1], s=2, alpha=0.05)
            plt.xlabel(r"$\theta_0$")
            plt.ylabel(r"$\theta_1$")
            plt.title("training data distribution")
            plt.savefig(f"{root}/{path}/training_data_{rd}.pdf")
            plt.close()

    if polyswyftSettings.plot_statistical_power:
        initial_size = polyswyftSettings.n_training_samples
        stats_power = np.empty(shape=(polyswyftSettings.NRE_num_retrain_rounds + 1,))
        for rd in range(0, polyswyftSettings.NRE_num_retrain_rounds + 1):
            samples = samples_storage[rd]
            if deadpoints_processing is not None:
                samples = deadpoints_processing(samples, rd)
            size = samples.shape[0]
            stats_power[rd] = size / initial_size
            initial_size += size

        plt.figure()
        plt.plot(stats_power)
        plt.xlabel("retrain round")
        plt.ylabel("num new samples / num training samples")
        plt.savefig(f"{root}/statistical_power.pdf")
        plt.close()
