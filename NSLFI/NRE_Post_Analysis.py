from typing import Dict

import matplotlib.pyplot as plt
import swyft
import torch
from anesthetic import MCMCSamples, make_2d_axes

from NSLFI.KL_divergence import compute_KL_divergence
from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NSNRE_data_generation import DataEnvironment


def plot_NRE_posterior(root_storage: Dict[str, str], network_storage: Dict[str, NRE_PolyChord],
                       nreSettings: NRE_Settings, dataEnv: DataEnvironment):
    # simulate full prior samples and compute true posterior
    prior_samples = dataEnv.sim.sample(
        nreSettings.n_weighted_samples * nreSettings.num_features * nreSettings.NRE_num_retrain_rounds)
    true_logLikes = torch.as_tensor(-prior_samples["l"])  # minus sign because of simulator convention
    true_samples = prior_samples[nreSettings.targetKey]
    weights_total = torch.exp(true_logLikes - true_logLikes.max()).sum()
    weights = torch.exp(true_logLikes - true_logLikes.max()) / weights_total * len(true_logLikes)
    weights = weights.numpy()

    # NRE refactoring
    prior_samples_nre = {nreSettings.targetKey: torch.as_tensor(prior_samples[nreSettings.targetKey])}
    obs = {nreSettings.obsKey: torch.tensor(dataEnv.obs[nreSettings.obsKey]).unsqueeze(0)}
    # set up labels for plotting
    params = [f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)]
    params_idx = [i for i in range(0, nreSettings.num_features)]
    params_labels = {i: f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)}

    # true posterior
    fig, axes = make_2d_axes(params_idx, labels=params_labels, lower=True, diagonal=True, upper=False, ticks="outer")
    mcmc_true = MCMCSamples(data=true_samples, logL=true_logLikes, weights=weights, labels=params_labels)
    mcmc_true.plot_2d(axes=axes, alpha=0.9, label="true", color="red",
                      kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})
    dkl_storage_true = []
    dkl_storage = []
    with torch.no_grad():
        # use trained NRE and evaluate on full prior samples
        for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
            network = network_storage[f"round_{rd}"]
            predictions = network.network(obs, prior_samples_nre)
            samples, weights = swyft.get_weighted_samples(predictions, params)
            logLs = predictions.logratios.numpy().squeeze()
            weights = weights.numpy().squeeze()
            samples = samples.numpy().squeeze()
            mcmc = MCMCSamples(data=samples, logL=logLs, weights=weights, labels=params_labels)
            KDL_true = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                             current_samples=mcmc_true.copy(), rd=rd + 1)
            dkl_storage_true.append(KDL_true)
            if rd != 0:
                KDL = compute_KL_divergence(nreSettings=nreSettings, network_storage=network_storage,
                                            current_samples=mcmc, rd=rd)
                dkl_storage.append(KDL)

            mcmc.plot_2d(axes=axes, alpha=0.4, label=f"rd {rd}", kinds={'lower': 'scatter_2d', 'diagonal': 'kde_1d'})
        root = root_storage["round_0"]
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center',
                                ncols=nreSettings.NRE_num_retrain_rounds + 2)
        fig.savefig(f"{root}/NRE_triangle_posterior.pdf")

        plt.figure()
        plt.errorbar(x=[x for x in range(1, nreSettings.NRE_num_retrain_rounds + 1)], y=[x[0] for x in dkl_storage],
                     yerr=[x[1] for x in dkl_storage],
                     label=r"$KL \mathrm{NRE}_i / \mathrm{NRE}_{i-1}$")
        plt.errorbar(x=[x for x in range(0, nreSettings.NRE_num_retrain_rounds + 1)],
                     y=[x[0] for x in dkl_storage_true],
                     yerr=[x[1] for x in dkl_storage_true], label=r"$KL \mathrm{True} / \mathrm{NRE}_i}$")
        plt.legend()
        plt.xlabel("round")
        plt.ylabel("KL divergence")
        plt.title("KL divergence between NRE rounds")
        plt.savefig(f"{root_storage['round_0']}/kl_divergence_truth.pdf")


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
