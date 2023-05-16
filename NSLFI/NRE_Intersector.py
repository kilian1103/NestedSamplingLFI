import logging
from typing import Tuple, Dict, Any

import torch
from torch import Tensor

from NSLFI.NRE_NS_Wrapper import NRE
from NSLFI.NRE_Settings import NRE_Settings


def intersect_samples(nreSettings: NRE_Settings, network_storage: Dict[str, NRE], root_storage: Dict[str, Any], rd: int,
                      boundarySample: Tensor) -> Tuple[Tensor, ...]:
    """Intersect samples from two NREs.
    :param nreSettings: NRE settings
    :param network_storage: dictionary containing the NREs
    :param root_storage: dictionary containing the root folders of the NREs
    :param obs: observation
    :param rd: round number
    :return: tuple of tensors containing the samples
    Do intersection of samples for two NREs. Where Left NRE is the NRE from the previous round and Right NRE is the
    NRE from the current round.
    """
    logger = logging.getLogger(nreSettings.logger_name)
    logger.info(f"intersecting samples using NRE {rd - 1} and {rd}")

    # load NREs
    previous_NRE_wrapped = network_storage[f"round_{rd - 1}"]
    previous_root = root_storage[f"round_{rd - 1}"]

    current_NRE_wrapped = network_storage[f"round_{rd}"]
    current_root = root_storage[f"round_{rd}"]

    # load samples
    previous_samples = torch.load(f=f"{previous_root}/posterior_samples")
    current_samples = torch.load(f=f"{current_root}/posterior_samples")

    # evaluate new contour using previous boundary sample
    previous_NRE_boundary_logL = previous_NRE_wrapped.logLikelihood(boundarySample)
    current_NRE_boundary_logL = current_NRE_wrapped.logLikelihood(boundarySample)

    if nreSettings.ns_nre_use_previous_boundary_sample_for_counting:

        previous_NRE_with_curr_samples_logLs = previous_NRE_wrapped.logLikelihood(current_samples)
        current_NRE_with_prev_samples_logLs = current_NRE_wrapped.logLikelihood(previous_samples)

        k1 = previous_samples[current_NRE_with_prev_samples_logLs > current_NRE_boundary_logL]
        l1 = previous_samples[current_NRE_with_prev_samples_logLs < current_NRE_boundary_logL]
        logger.info(f"k1 round {rd - 1} samples within NRE_{rd} using NRE_{rd - 1} boundary sample: {len(k1)}")
        logger.info(f"l1 round {rd - 1} samples outside NRE_{rd} using NRE_{rd - 1}  boundary sample: {len(l1)}")

        k2 = current_samples[previous_NRE_with_curr_samples_logLs > previous_NRE_boundary_logL]
        l2 = current_samples[previous_NRE_with_curr_samples_logLs < previous_NRE_boundary_logL]
        logger.info(f"k2 round {rd} samples within NRE_{rd - 1} using NRE_{rd - 1} boundary sample: {len(k2)}")
        logger.info(f"l2 round {rd} samples outside NRE_{rd - 1} using NRE_{rd - 1} boundary sample: {len(l2)}")

    else:
        # use current NRE boundary sample for counting
        # evaluate logLs of previous samples
        previous_NRE_with_prev_samples_logLs = previous_NRE_wrapped.logLikelihood(previous_samples)
        previous_NRE_with_curr_samples_logLs = previous_NRE_wrapped.logLikelihood(current_samples)

        # compute contraction of previous NRE contour when choosing a boundary sample from current NRE
        n1 = previous_samples[previous_NRE_with_prev_samples_logLs > previous_NRE_boundary_logL]
        n0 = previous_samples[previous_NRE_with_prev_samples_logLs < previous_NRE_boundary_logL]
        contraction = len(n1) / (len(n0) + len(n1))  # of prev NRE contour by choosing curr NRE boundary sample
        logger.info(f"n1 round {rd - 1} samples within NRE_{rd - 1} using NRE_{rd} boundary sample: {len(n1)}")
        logger.info(f"n0 round {rd - 1} samples outside NRE_{rd - 1} using NRE_{rd} boundary sample: {len(n0)}")
        logger.info(f"contraction factor of NRE_{rd - 1} area using NRE_{rd} boundary sample: {contraction}")

        # count number of contracted space samples satisfying current NRE contour
        current_NRE_with_prev_contracted_space_samples_logLs = current_NRE_wrapped.logLikelihood(n1)
        k1 = n1[current_NRE_with_prev_contracted_space_samples_logLs > current_NRE_boundary_logL]
        l1 = n1[current_NRE_with_prev_contracted_space_samples_logLs < current_NRE_boundary_logL]
        logger.info(f"k1 round {rd - 1} samples within NRE_{rd} using NRE_{rd} boundary sample: {len(k1)}")
        logger.info(f"l1 round {rd - 1} samples outside NRE_{rd} using NRE_{rd} boundary sample: {len(l1)}")

        # count number of current samples satisfying previous NRE contour
        k2 = current_samples[previous_NRE_with_curr_samples_logLs > previous_NRE_boundary_logL]
        l2 = current_samples[previous_NRE_with_curr_samples_logLs < previous_NRE_boundary_logL]
        logger.info(f"k2 round {rd} samples within NRE_{rd - 1} using NRE_{rd} boundary sample: {len(k2)}")
        logger.info(f"l2 round {rd} samples outside NRE_{rd - 1} using NRE_{rd} boundary sample: {len(l2)}")

    torch.save(obj=k1, f=f"{previous_root}/k1")
    torch.save(obj=l1, f=f"{previous_root}/l1")
    torch.save(obj=k2, f=f"{previous_root}/k2")
    torch.save(obj=l2, f=f"{previous_root}/l2")
    if nreSettings.ns_nre_use_previous_boundary_sample_for_counting:
        return k1, l1, k2, l2
    else:
        torch.save(obj=n1, f=f"{previous_root}/n1")
        torch.save(obj=n0, f=f"{previous_root}/n0")
        return k1, l1, k2, l2, n0, n1
