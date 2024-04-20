from typing import Dict, Tuple

import anesthetic
import matplotlib.pyplot as plt
import swyft
from anesthetic import make_2d_axes, make_1d_axes

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.utils import compute_KL_compression, compute_KL_divergence_truth


def plot_analysis_of_NSNRE(root: str, network_storage: Dict[int, swyft.SwyftModule],
                           samples_storage: Dict[int, anesthetic.Samples], dkl_storage: Dict[int, Tuple[float, float]],
                           nreSettings: NRE_Settings,
                           obs: swyft.Sample, true_posterior: anesthetic.Samples = None):
    """
    Plot the analysis of the NSNRE.
    :param root: A string of the root directory to save the plots
    :param root_storage: A dictionary of roots for each round
    :param network_storage: A dictionary of networks for each round
    :param samples_storage: A dictionary of samples for each round
    :param dkl_storage: A dictionary of KL divergences for each round
    :param nreSettings: A NRE_Settings object
    :param obs: A Swyft sample of the observed data
    :param true_posterior: An anesthetic samples object of the true posterior if available
    :return:
    """
    # set up labels for plotting
    params_idx = [i for i in range(0, nreSettings.num_features)]
    params_labels = {i: rf"${nreSettings.targetKey}_{i}$" for i in range(nreSettings.num_features)}

    dkl_storage_true = {}

    # triangle plot
    if nreSettings.plot_triangle_plot:
        kinds = {'lower': 'kde_2d', 'diagonal': 'kde_1d', 'upper': "scatter_2d"}
        fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=True,
                                 ticks="outer")
        last_round_samples = samples_storage[nreSettings.NRE_num_retrain_rounds]
        # load prior from last round
        prior = last_round_samples.prior()
        prior.plot_2d(axes=axes, alpha=0.4, label="prior", kinds=kinds)
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            nested.plot_2d(axes=axes, alpha=0.4, label=fr"$p(\theta|D)_{rd}$",
                           kinds=kinds)
        if true_posterior is not None:
            true_posterior.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                   kinds=kinds)

        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_full.pdf")

        ### zoomed in triangle plot
        fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=True,
                                 ticks="outer")
        for rd in range(nreSettings.triangle_zoom_start,
                        nreSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            nested.plot_2d(axes=axes, alpha=0.4, label=fr"$p(\theta|D)_{rd}$",
                           kinds=kinds)
        if true_posterior is not None:
            true_posterior.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                   kinds=kinds)
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_zoom.pdf")

    # KL divergence plot
    if nreSettings.plot_KL_divergence:
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            if true_posterior is not None:
                previous_network = network_storage[rd]
                KDL_true = compute_KL_divergence_truth(nreSettings=nreSettings,
                                                       previous_network=previous_network.eval(),
                                                       true_posterior=true_posterior.copy(), obs=obs,
                                                       previous_samples=samples_storage[rd])
                dkl_storage_true[rd] = KDL_true
        plt.figure()
        plt.errorbar(x=[i for i in range(1, nreSettings.NRE_num_retrain_rounds + 1)],
                     y=[dkl_storage[i][0] for i in range(1, nreSettings.NRE_num_retrain_rounds + 1)],
                     yerr=[dkl_storage[i][1] for i in range(1, nreSettings.NRE_num_retrain_rounds + 1)],
                     label=r"$KL \mathrm{NRE}_i / \mathrm{NRE}_{i-1}, corr$")
        if true_posterior is not None:
            plt.errorbar(x=[i for i in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                         y=[dkl_storage_true[i][0] for i in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                         yerr=[dkl_storage_true[i][1] for i in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                         label=r"$KL \mathrm{True} / \mathrm{NRE}_i}$")
        plt.legend()
        plt.xlabel("round")
        plt.ylabel("KL divergence")
        plt.title("KL divergence between NRE rounds")
        plt.savefig(f"{root}/kl_divergence.pdf")

    if nreSettings.plot_KL_compression:
        dkl_compression_storage = {}
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            DKL = compute_KL_compression(samples_storage[rd], nreSettings)
            dkl_compression_storage[rd] = DKL

        plt.figure()
        plt.errorbar(x=[i for i in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                     y=[dkl_compression_storage[i][0] for i in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                     yerr=[dkl_compression_storage[i][1] for i in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                     label=r"$KL(P||\pi)$")
        plt.legend()
        plt.xlabel("round")
        plt.ylabel("DKL")
        plt.title("DKL compression of Prior to Posterior")
        plt.savefig(f"{root}/kl_compression.pdf")

    if nreSettings.plot_logR_histogram:
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            samples = samples_storage[rd]
            logRs = samples["logL"] - samples.logZ()
            plt.hist(logRs, label=f"round {rd}", alpha=0.5)
        plt.title(r"$\log r$ histogram")
        plt.xlabel(r"$\log r$")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"{root}/logR_histogram_unweighted.pdf")

    if nreSettings.plot_logR_pdf:
        figs, axes = make_1d_axes("logR", figsize=(3.5, 3.5))
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            samples = samples_storage[rd]
            samples["logR"] = samples["logL"] - samples.logZ()
            samples.plot_1d(axes=axes, label=f"round {rd}")
        plt.xlabel(r"$\log r$")
        plt.ylabel(r"$p(\log r)$")
        plt.legend()
        plt.savefig(f"{root}/logR_pdf.pdf", dpi=300, bbox_inches='tight')
