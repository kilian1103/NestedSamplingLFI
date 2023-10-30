from typing import Dict

import anesthetic
import matplotlib.pyplot as plt
import swyft
import torch
from anesthetic import MCMCSamples, make_2d_axes, read_chains

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.utils import compute_KL_divergence


def plot_analysis_of_NSNRE(root_storage: Dict[str, str], network_storage: Dict[str, swyft.SwyftModule],
                           nreSettings: NRE_Settings, sim: swyft.Simulator, obs: swyft.Sample):
    # set up labels for plotting
    params = [f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)]
    params_idx = [i for i in range(0, nreSettings.num_features)]
    params_labels = {i: rf"${nreSettings.targetKey}_{i}$" for i in range(nreSettings.num_features)}

    params_labels_ext = params_labels.copy()
    params_labels_ext.update({nreSettings.num_features + j: rf"$D_{j}$" for j in
                              range(nreSettings.num_features_dataset)})
    params_idx_ext = [i for i in range(0, nreSettings.num_features + nreSettings.num_features_dataset)]

    samples_storage = []
    dkl_storage = []
    root = root_storage[f"round_{nreSettings.NRE_num_retrain_rounds}"]

    # load data for plots
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        samples = anesthetic.read_chains(root=f"{root_storage[f'round_{rd}']}/{nreSettings.file_root}")
        samples_storage.append(samples)

    if nreSettings.true_contours_available:
        dkl_storage_true = []
        full_joint = sim.sample(nreSettings.n_compressed_weighted_samples)
        true_logLikes = torch.as_tensor(full_joint[nreSettings.contourKey])
        posterior = full_joint[nreSettings.posteriorsKey]
        posterior_theta = torch.empty(size=(len(posterior), nreSettings.num_features))
        weights = torch.empty(size=(len(posterior), 1))
        for i, sample in enumerate(posterior):
            theta, weight = sample
            posterior_theta[i, :] = theta
            weights[i] = weight

        mcmc_true = MCMCSamples(
            data=posterior_theta, weights=weights.squeeze(),
            logL=true_logLikes, labels=params_labels)

    # triangle plot
    if nreSettings.plot_triangle_plot:
        fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=False,
                                 ticks="outer")
        if nreSettings.true_contours_available:
            mcmc_true.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                              kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})

        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            nested.plot_2d(axes=axes, alpha=0.4, label=f"rd {rd}",
                           kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior.pdf")

    # data and param triangle plot
    if nreSettings.plot_triangle_plot_ext:
        fig, axes = make_2d_axes(params_idx_ext, labels=params_labels_ext, lower=True, diagonal=True, upper=False,
                                 ticks="outer")
        if nreSettings.true_contours_available:
            # TODO fix this
            mcmc_true_ext = MCMCSamples(
                data=torch.cat(
                    (torch.as_tensor(full_joint[nreSettings.targetKey]),
                     torch.as_tensor(full_joint[nreSettings.obsKey])), dim=1),
                logL=true_logLikes,
                weights=weights, labels=params_labels_ext)
            mcmc_true_ext = mcmc_true_ext.compress(nreSettings.n_compressed_weighted_samples).drop_duplicates()
            mcmc_true_ext.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                  kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})

        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            nested.plot_2d(axes=axes, alpha=0.4, label=f"rd {rd}",
                           kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_ext.pdf")

    # KL divergence plot
    if nreSettings.plot_KL_divergence:
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            if nreSettings.true_contours_available:
                KDL_true = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                                 current_samples=mcmc_true.copy(), rd=rd + 1, obs=obs)
                dkl_storage_true.append(KDL_true)
            if rd != 0:
                nested = samples_storage[rd]
                KDL = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                            current_samples=nested, rd=rd, obs=obs)
                dkl_storage.append(KDL)
        plt.figure()
        plt.errorbar(x=[x for x in range(1, nreSettings.NRE_num_retrain_rounds + 1)], y=[x[0] for x in dkl_storage],
                     yerr=[x[1] for x in dkl_storage],
                     label=r"$KL \mathrm{NRE}_i / \mathrm{NRE}_{i-1}$")
        if nreSettings.true_contours_available:
            plt.errorbar(x=[x for x in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                         y=[x[0] for x in dkl_storage_true],
                         yerr=[x[1] for x in dkl_storage_true], label=r"$KL \mathrm{True} / \mathrm{NRE}_i}$")
        plt.legend()
        plt.xlabel("round")
        plt.ylabel("KL divergence")
        plt.title("KL divergence between NRE rounds")
        plt.savefig(f"{root}/kl_divergence.pdf")

    # plot quantile plot of training dataset
    if nreSettings.plot_quantile_plot:
        for i in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            root = root_storage[f"round_{i}"]
            samples = read_chains(root=f"{root}/{nreSettings.file_root}")
            samples = samples.iloc[:, :nreSettings.num_features]
            plot_quantile_plot(samples=samples, nreSettings=nreSettings, root=root)


def plot_quantile_plot(samples, nreSettings: NRE_Settings, root: str):
    samples = samples.drop_weights()
    quantiles = samples.quantile(nreSettings.percentiles_of_quantile_plot)
    plt.figure()
    for i in range(nreSettings.num_features):
        plt.plot(nreSettings.percentiles_of_quantile_plot, quantiles.iloc[:, i], label=rf"$z_{i}$")
    plt.xlabel("quantile")
    plt.ylabel(rf"$z$ value")
    plt.title("Quantile plot of training dataset")
    plt.legend()
    plt.savefig(f"{root}/quantile_plot.pdf")


def plot_NRE_expansion_and_contraction_rate(root_storage: Dict[str, str], nreSettings: NRE_Settings):
    data_exp = []
    data_comp = []
    data_rate = []
    for rd in range(0, nreSettings.NRE_num_retrain_rounds):
        root = root_storage[f"round_{rd}"]
        k1 = torch.load(f"{root}/k1")
        l1 = torch.load(f"{root}/l1")
        k2 = torch.load(f"{root}/k2")
        l2 = torch.load(f"{root}/l2")
        comp = len(k1) / (len(k1) + len(l1))
        data_comp.append(comp)
        expan = (len(k2) + len(l2)) / len(k2)
        data_exp.append(expan)
        rate = comp * expan
        data_rate.append(rate)

    iter = list(range(1, nreSettings.NRE_num_retrain_rounds + 1))
    plt.figure()
    plt.plot(iter, data_comp, label="compression")
    plt.plot(iter, data_exp, label="expansion")
    plt.plot(iter, data_rate, label="total rate")
    plt.xlabel("round")
    plt.ylabel("rate")
    plt.legend()
    plt.savefig(f"{root_storage[f'round_{nreSettings.NRE_num_retrain_rounds}']}/NRE_expansion_and_contraction_rate.pdf")
