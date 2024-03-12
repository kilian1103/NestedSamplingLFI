from typing import Dict

import anesthetic
import matplotlib.pyplot as plt
import numpy as np
import swyft
import torch
from anesthetic import MCMCSamples, make_2d_axes
from pypolychord import PolyChordSettings
from swyft import collate_output as reformat_samples

from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.utils import compute_KL_divergence, select_weighted_contour, compute_KL_compression


def plot_analysis_of_NSNRE(root_storage: Dict[int, str], network_storage: Dict[int, swyft.SwyftModule],
                           nreSettings: NRE_Settings, polyset: PolyChordSettings, sim: swyft.Simulator,
                           obs: swyft.Sample):
    # set up labels for plotting
    params = [f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)]
    params_idx = [i for i in range(0, nreSettings.num_features)]
    params_labels = {i: rf"${nreSettings.targetKey}_{i}$" for i in range(nreSettings.num_features)}

    params_labels_ext = params_labels.copy()
    params_labels_ext.update(
        {nreSettings.num_features + j: rf"$D_{j}$" for j in range(nreSettings.num_features_dataset)})
    params_idx_ext = [i for i in range(0, nreSettings.num_features + nreSettings.num_features_dataset)]
    samples_storage = []
    dkl_storage = []
    root = root_storage[nreSettings.NRE_num_retrain_rounds]

    # load data for plots
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        if nreSettings.use_livepoint_increasing:
            samples = anesthetic.read_chains(
                root=f"{root_storage[rd]}/{nreSettings.increased_livepoints_fileroot}/{polyset.file_root}")
        else:
            samples = anesthetic.read_chains(root=f"{root_storage[rd]}/{polyset.file_root}")

        if nreSettings.use_dataset_clipping:
            index = select_weighted_contour(samples, nreSettings.dataset_posterior_clipping_contour)
            samples = samples.truncate(index)

        samples_storage.append(samples.copy())

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

    # data and param triangle plot
    if nreSettings.plot_triangle_plot_ext:
        kinds = {'lower': 'scatter_2d', 'diagonal': 'kde_1d'}
        fig, axes = make_2d_axes(params_idx_ext, labels=params_labels_ext, lower=True, diagonal=True, upper=False,
                                 ticks="outer")
        if nreSettings.true_contours_available:
            # TODO fix this
            mcmc_true_ext = MCMCSamples(
                data=torch.cat(
                    (torch.as_tensor(full_joint[nreSettings.targetKey]),
                     torch.as_tensor(full_joint[nreSettings.obsKey])), dim=1),
                logL=true_logratios,
                weights=weights, labels=params_labels_ext)
            mcmc_true_ext.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                  kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})

        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            nested = samples_storage[rd]
            theta = nested.iloc[:, :nreSettings.num_features]
            theta = torch.as_tensor(theta.to_numpy())
            joints = []
            for point in theta:
                cond = {nreSettings.targetKey: point.float()}
                sample = sim.sample(conditions=cond, targets=[nreSettings.obsKey])
                joints.append(sample)
            joints = reformat_samples(joints)
            joints = joints[nreSettings.obsKey]
            for nd in range(nreSettings.num_features, nreSettings.num_features + nreSettings.num_features_dataset):
                nested[nd] = joints[:, nd - nreSettings.num_features]
            nested.plot_2d(axes=axes, alpha=0.4, label=f"rd_{rd}", kinds=kinds)

        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_ext.pdf")

    # KL divergence plot
    if nreSettings.plot_KL_divergence:
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            if nreSettings.true_contours_available:
                previous_network = network_storage[rd]
                KDL_true = compute_KL_divergence(nreSettings=nreSettings, previous_network=previous_network.eval(),
                                                 current_samples=mcmc_true.copy(), obs=obs,
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
        plt.savefig("logR_histogram.pdf")


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
