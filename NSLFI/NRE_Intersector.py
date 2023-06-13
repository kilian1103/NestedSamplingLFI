import logging
from typing import Tuple, Dict, Any

import torch
from torch import Tensor

from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings


def intersect_samples(nreSettings: NRE_Settings, network_storage: Dict[str, NRE_PolyChord],
                      root_storage: Dict[str, Any], rd: int,
                      boundarySample: Tensor, previous_samples: Tensor, current_samples: Tensor) -> Tuple[Tensor, ...]:
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
    with torch.no_grad():
        # evaluate new contour using previous boundary sample
        previous_NRE_boundary_logL = previous_NRE_wrapped.logLikelihood(boundarySample)
        current_NRE_boundary_logL = current_NRE_wrapped.logLikelihood(boundarySample)

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

    torch.save(obj=k1, f=f"{previous_root}/k1")
    torch.save(obj=l1, f=f"{previous_root}/l1")
    torch.save(obj=k2, f=f"{previous_root}/k2")
    torch.save(obj=l2, f=f"{previous_root}/l2")
    return k1, l1, k2, l2
