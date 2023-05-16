import os
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from NSLFI.NRE_Intersector import intersect_samples
from NSLFI.NRE_Settings import NRE_Settings


class nre_simulator:
    def __init__(self, loglike: Any):
        self.loglike = loglike

    def logLikelihood(self, proposal_sample: Tensor) -> Tensor:
        return self.loglike(proposal_sample)


def L1(x: Tensor) -> Tensor:
    circle_mid = torch.tensor([0, 0])
    return -torch.square(x - circle_mid).sum(dim=-1)


def L2(x: Tensor) -> Tensor:
    circle_mid = torch.tensor([0.3, 0])
    return -torch.square(x - circle_mid).sum(dim=-1)


def gen(l, ls, n):
    points, likes = [], []
    while len(points) < n:
        x = torch.randn(2)
        l_ = l(x)
        if l_ > ls and (x > -1).all() and (x < 1).all():
            points.append(x)
            likes.append(l_)

    i = torch.argsort(torch.stack(likes))
    return torch.stack(points)[i], torch.stack(likes)[i]


def plot_contour(ax, l, *args, **kwargs):
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    z = np.array([l(torch.tensor([x_, y_])) for x_, y_ in zip(x.flatten(), y.flatten())]).reshape(x.shape)
    return ax.contour(x, y, z, *args, **kwargs)


def test_intersect_samples_prev_bound():
    # initialization
    n0 = n1 = 100
    boundarySample = torch.tensor([-0.2, 0])  # circle should be contained fully within other circle

    # setup settings
    nreSettings = NRE_Settings()
    nreSettings.ns_nre_use_previous_boundary_sample_for_counting = True

    # setup root directory
    nre_0_root = "nre_0_root"
    nre_1_root = "nre_1_root"
    try:
        os.makedirs(nre_0_root, exist_ok=True)
        os.makedirs(nre_1_root, exist_ok=True)
    except FileExistsError:
        pass

    # setup samples to count
    minLog0 = L1(boundarySample)
    minLog1 = L2(boundarySample)
    posterior_0, likes_0 = gen(L1, minLog0, n0)
    posterior_1, likes_1 = gen(L2, minLog1, n1)

    fig, ax = plt.subplots()
    ax.plot(*boundarySample, 'x', label=r'$\theta_Y$', ms=10)
    CS1 = plot_contour(ax, L1, levels=[minLog0], colors=['C0'], linestyles='solid')
    ax.plot(*posterior_0.T, 'C0.', label='$X$')

    CS2 = plot_contour(ax, L2, levels=[minLog1], colors=['C1'], linestyles='solid')
    ax.plot(*posterior_1.T, 'C1.', label='$Z$')

    handles, labels = ax.get_legend_handles_labels()
    h, l = CS2.legend_elements()
    handles = [h[0]] + handles
    labels = [r'$\mathcal{L}^{(2)}(\theta_Y)$'] + labels
    h, l = CS1.legend_elements()
    handles = [h[0]] + handles
    labels = [r'$\mathcal{L}^{(1)}(\theta_Y)$'] + labels

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)
    ax.legend(handles, labels)
    ax.set_aspect(1)
    fig.tight_layout()
    plt.show()

    # save samples in correct directory
    torch.save(obj=posterior_0, f=os.path.join(nre_0_root, "posterior_samples"))
    torch.save(obj=posterior_1, f=os.path.join(nre_1_root, "posterior_samples"))
    root_storage = {"round_0": nre_0_root,
                    "round_1": nre_1_root}

    # setup NRE loglike functions
    nre_0 = nre_simulator(loglike=L1)
    nre_1 = nre_simulator(loglike=L2)
    network_storage = {"round_0": nre_0, "round_1": nre_1}

    # simulate first round
    rd = 1
    k1, l1, k2, l2 = intersect_samples(nreSettings=nreSettings, root_storage=root_storage,
                                       network_storage=network_storage, rd=rd,
                                       boundarySample=boundarySample)
    os.removedirs(nre_0_root)
    os.removedirs(nre_1_root)

    pass
