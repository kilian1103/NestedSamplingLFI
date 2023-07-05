from typing import Dict

import matplotlib.pyplot as plt
import swyft
import torch
from anesthetic import MCMCSamples, make_2d_axes

from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings
from NSLFI.NSNRE_data_generation import DataEnvironment


def plot_NRE_posterior(root_storage: Dict[str, str], network_storage: Dict[str, NRE_PolyChord],
                       nreSettings: NRE_Settings, dataEnv: DataEnvironment):
    # simulate full prior samples and compute true posterior
    prior_samples = dataEnv.sim.sample(nreSettings.n_weighted_samples)
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
    params_idx = [i for i in range(nreSettings.num_features - 1, -1, -1)]
    params_labels = {i: f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)}

    # true posterior
    fig, axes = make_2d_axes(params_idx, labels=params_labels)
    mcmc = MCMCSamples(data=true_samples, logL=true_logLikes, weights=weights, labels=params_labels)
    mcmc.plot_2d(axes=axes, alpha=0.9, label="true contours", color="red")

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
            mcmc.plot_2d(axes=axes, alpha=0.9, label=f"round {rd}")
        root = root_storage["round_0"]
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center', ncols=2)
        fig.savefig(f"{root}/NRE_triangle_posterior.pdf")


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
