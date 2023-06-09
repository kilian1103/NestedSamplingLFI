from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from getdist import MCSamples, plots

from NSLFI.NRE_Settings import NRE_Settings


def plot_NRE_posterior(root_storage: Dict[str, str], nreSettings: NRE_Settings):
    data = []
    g = plots.getSubplotPlotter(width_inch=7.055)
    g.settings.x_label_rotation = 45  # This stops the x axis labels
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        root = root_storage[f"round_{rd}"]
        dat = np.loadtxt(f"{root}/{nreSettings.file_root}.txt")
        weights = dat[-nreSettings.n_training_samples:, 0]
        logL = dat[-nreSettings.n_training_samples:, 1]
        samples = dat[-nreSettings.n_training_samples:, 2:]
        sample = MCSamples(samples=samples, weights=weights, loglikes=logL, label=f"round {rd}")
        data.append(sample)
    g.triangle_plot(data, filled=True,
                    labels=[f"{nreSettings.targetKey}[{i}]" for i in range(nreSettings.num_features)])
    root = root_storage["round_0"]
    g.export(f"{root}/NRE_triangle_plot.pdf")


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
