import torch
from getdist import MCSamples, plots

from NSLFI.NRE_Settings import NRE_Settings


def plot_NRE_posterior(root_storage, nreSettings: NRE_Settings):
    data = []
    g = plots.getSubplotPlotter(width_inch=7.055)
    g.settings.x_label_rotation = 45  # This stops the x axis labels
    for rd in range(0, nreSettings.NRE_num_retrain_rounds + 1):
        root = root_storage[f"round_{rd}"]
        samples = torch.load(f=f"{root}/posterior_samples")
        weights = torch.load(f=f"{root}/logL")
        sample = MCSamples(samples=samples.numpy(), weights=weights.numpy(), label=f"round {rd + 1}")
        data.append(sample)
    g.triangle_plot(data, filled=True, labels=["z[0]", "z[1]"])
    g.export(f"NRE_triangle_plot.pdf")
