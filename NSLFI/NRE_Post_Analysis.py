from typing import Dict, Tuple

import anesthetic
import matplotlib.pyplot as plt
import numpy as np
import swyft
import torch
from anesthetic import MCMCSamples, make_2d_axes, make_1d_axes

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.utils import compute_KL_divergence, compute_KL_compression


def plot_analysis_of_NSNRE(root_storage: Dict[int, str], network_storage: Dict[int, swyft.SwyftModule],
                           samples_storage: Dict[int, anesthetic.Samples],
                           nreSettings: NRE_Settings, sim: swyft.Simulator,
                           obs: swyft.Sample):
    # set up labels for plotting
    params_idx = [i for i in range(0, nreSettings.num_features)]
    params_labels = {i: rf"${nreSettings.targetKey}_{i}$" for i in range(nreSettings.num_features)}
    params_labels_ext = params_labels.copy()
    params_labels_ext.update(
        {nreSettings.num_features + j: rf"$D_{j}$" for j in range(nreSettings.num_features_dataset)})
    params_idx_ext = [i for i in range(0, nreSettings.num_features + nreSettings.num_features_dataset)]

    dkl_storage = []
    root = root_storage[nreSettings.NRE_num_retrain_rounds]

    if nreSettings.true_contours_available:
        dkl_storage_true = []
        cond = {nreSettings.obsKey: obs[nreSettings.obsKey].numpy().squeeze()}
        full_joint = sim.sample(nreSettings.n_weighted_samples, conditions=cond)
        true_logratios = torch.as_tensor(full_joint[nreSettings.contourKey])
        posterior = full_joint[nreSettings.posteriorsKey]
        weights = np.ones(shape=len(posterior))  # direct samples from posterior have weights 1

        mcmc_true = MCMCSamples(
            data=posterior, weights=weights.squeeze(),
            logL=true_logratios, labels=params_labels)

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
        if nreSettings.true_contours_available:
            mcmc_true.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
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
        if nreSettings.true_contours_available:
            mcmc_true.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                              kinds=kinds)
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_zoom.pdf")

    # KL divergence plot
    if nreSettings.plot_KL_divergence:
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            if nreSettings.true_contours_available:
                previous_network = network_storage[rd]
                KDL_true = compute_KL_divergence_truth(nreSettings=nreSettings,
                                                       previous_network=previous_network.eval(),
                                                       true_posterior=mcmc_true.copy(), obs=obs,
                                                       previous_samples=samples_storage[rd])
                dkl_storage_true.append(KDL_true)
            if rd != 0:
                nested = samples_storage[rd]
                previous_network = network_storage[rd - 1]
                KDL = compute_KL_divergence(nreSettings=nreSettings, previous_network=previous_network.eval(),
                                            current_samples=nested, obs=obs, previous_samples=samples_storage[rd - 1])
                dkl_storage.append(KDL)
        plt.figure()
        plt.errorbar(x=[x for x in range(1, nreSettings.NRE_num_retrain_rounds + 1)], y=[x[0] for x in dkl_storage],
                     yerr=[x[1] for x in dkl_storage],
                     label=r"$KL \mathrm{NRE}_i / \mathrm{NRE}_{i-1}, corr$")
        if nreSettings.true_contours_available:
            plt.errorbar(x=[x for x in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                         y=[x[0] for x in dkl_storage_true],
                         yerr=[x[1] for x in dkl_storage_true], label=r"$KL \mathrm{True} / \mathrm{NRE}_i}$")
        plt.legend()
        plt.xlabel("round")
        plt.ylabel("KL divergence")
        plt.title("KL divergence between NRE rounds")
        plt.savefig(f"{root}/kl_divergence.pdf")

    if nreSettings.plot_KL_compression:
        DKL_storage = []
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            DKL = compute_KL_compression(samples_storage[rd], nreSettings)
            DKL_storage.append(DKL)

        plt.figure()
        plt.errorbar(x=[x for x in range(0, nreSettings.NRE_num_retrain_rounds + 1)], y=[x[0] for x in DKL_storage],
                     yerr=[x[1] for x in DKL_storage],
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
        plt.savefig("logR_histogram_unweighted.pdf")

    if nreSettings.plot_logR_pdf:
        figs, axes = make_1d_axes("logR", figsize=(3.5, 3.5))
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            samples = samples_storage[rd]
            samples["logR"] = samples["logL"] - samples.logZ()
            samples.plot_1d(axes=axes, label=f"round {rd}")
        plt.xlabel(r"$\log r$")
        plt.ylabel(r"$p(\log r)$")
        plt.legend()
        plt.savefig("logR_pdf.pdf", dpi=300, bbox_inches='tight')


def compute_KL_divergence_truth(nreSettings: NRE_Settings, previous_network: swyft.SwyftModule,
                                true_posterior: anesthetic.Samples, previous_samples: anesthetic.Samples,
                                obs: swyft.Sample) -> Tuple[float, float]:
    swyft_samples = {
        nreSettings.targetKey: torch.as_tensor(true_posterior.iloc[:, :nreSettings.num_features].to_numpy())}
    with torch.no_grad():
        predictions = previous_network(obs, swyft_samples)
    true_posterior["logL_previous"] = predictions.logratios.numpy().squeeze()
    # MCMC samples for true samples do not have logw functionality
    samples = true_posterior.iloc[:, :nreSettings.num_features].squeeze()
    true_posterior_logL = nreSettings.model.posterior(obs[nreSettings.obsKey].numpy().squeeze()).logpdf(samples)
    true_prior = nreSettings.model.prior().logpdf(samples)
    true_posterior.logL = true_posterior_logL
    true_posterior["logR"] = true_posterior["logL_previous"]
    logpqs = (true_posterior["logL"].values[:, None] - true_posterior["logR"].values[:, None] - true_prior[:,
                                                                                                None] +
              previous_samples.logZ(
                  nreSettings.n_DKL_estimates).values)
    DKL_estimates = logpqs.mean(axis=0)
    DKL = DKL_estimates.mean()
    DKL_err = DKL_estimates.std()
    return (DKL, DKL_err)
