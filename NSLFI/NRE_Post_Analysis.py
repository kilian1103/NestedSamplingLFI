from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import swyft
import torch
from anesthetic import MCMCSamples, make_2d_axes
from swyft import collate_output as reformat_samples

from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NSNRE_data_generation import DataEnvironment
from NSLFI.utils import compute_KL_divergence


def plot_NRE_posterior(root_storage: Dict[str, str], network_storage: Dict[str, NRE_PolyChord],
                       nreSettings: NRE_Settings, dataEnv: DataEnvironment):
    # full prior samples of NSNRE
    theta_samples = np.random.uniform(nreSettings.sim_prior_lower,
                                      nreSettings.sim_prior_lower + nreSettings.prior_width,
                                      (nreSettings.n_weighted_samples, nreSettings.num_features))
    joints = []
    for theta in theta_samples:
        cond = {nreSettings.targetKey: theta}
        joint = dataEnv.sim.sample(conditions=cond)
        joints.append(joint)
    joints = reformat_samples(joints)
    data_samples = joints[nreSettings.obsKey]
    # NRE refactoring
    prior_samples_nre = {nreSettings.targetKey: torch.as_tensor(theta_samples)}
    obs = {nreSettings.obsKey: torch.tensor(dataEnv.obs[nreSettings.obsKey]).unsqueeze(0)}

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
    root = root_storage["round_0"]

    if nreSettings.true_contours_available:
        dkl_storage_true = []
        true_logLikes = torch.as_tensor(
            -joints[nreSettings.contourKey])  # minus sign because of simulator convention
        # true posterior
        weights_total = torch.exp(true_logLikes - true_logLikes.max()).sum()
        weights = torch.exp(true_logLikes - true_logLikes.max()) / weights_total * len(true_logLikes)
        weights = weights.numpy()

        mcmc_true = MCMCSamples(data=joints[nreSettings.targetKey], logL=true_logLikes, weights=weights,
                                labels=params_labels)
        mcmc_true.compress()

    # load data for plots
    with torch.no_grad():
        # use trained NRE and evaluate on full prior samples
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            network = network_storage[f"round_{rd}"]
            predictions = network.network(obs, prior_samples_nre)
            samples, weights = swyft.get_weighted_samples(predictions, params)
            logLs = predictions.logratios.numpy().squeeze()
            weights = weights.numpy().squeeze()
            samples = samples.numpy().squeeze()
            mcmc = MCMCSamples(data=torch.cat((torch.as_tensor(samples), torch.as_tensor(data_samples)), dim=1),
                               logL=logLs,
                               weights=weights,
                               labels=params_labels_ext)
            mcmc.compress()
            samples_storage.append(mcmc)

    # triangle plot
    if nreSettings.plot_triangle_plot:
        fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=False,
                                 ticks="outer")
        if nreSettings.true_contours_available:
            mcmc_true.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                              kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})

        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            mcmc = samples_storage[rd]
            mcmc.plot_2d(axes=axes, alpha=0.4, label=f"rd {rd}",
                         kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior.pdf")

    # data and param triangle plot
    if nreSettings.plot_triangle_plot_ext:

        fig, axes = make_2d_axes(params_idx_ext, labels=params_labels_ext, lower=True, diagonal=True, upper=False,
                                 ticks="outer")
        if nreSettings.true_contours_available:
            mcmc_true_ext = MCMCSamples(
                data=torch.cat((joints[nreSettings.targetKey], joints[nreSettings.obsKey]), dim=1),
                logL=true_logLikes,
                weights=weights, labels=params_labels_ext)
            mcmc_true_ext.compress()
            mcmc_true_ext.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                                  kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})

        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            mcmc = samples_storage[rd]
            mcmc.plot_2d(axes=axes, alpha=0.4, label=f"rd {rd}",
                         kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior_ext.pdf")

    # KL divergence plot
    if nreSettings.plot_KL_divergence:
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            if nreSettings.true_contours_available:
                KDL_true = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                                 current_samples=mcmc_true.copy(), rd=rd + 1)
                dkl_storage_true.append(KDL_true)
            if rd != 0:
                mcmc = samples_storage[rd]
                KDL = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                            current_samples=mcmc, rd=rd)
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
        plt.savefig(f"{root}/kl_divergence_truth.pdf")


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
    plt.savefig(f"{root_storage['round_0']}/NRE_expansion_and_contraction_rate.pdf")
