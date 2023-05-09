import logging
from typing import Tuple, Dict, Any

import swyft
import torch
from torch import Tensor

from NSLFI.NRE_NS_Wrapper import NRE
from NSLFI.NRE_Settings import NRE_Settings


def intersect_samples(nreSettings: NRE_Settings, network_storage: Dict[str, Any], root_storage: Dict[str, Any],
                      obs: swyft.Sample, rd: int) -> Tuple[Tensor, ...]:
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
    previous_NRE = network_storage[f"round_{rd - 1}"]
    previous_NRE_wrapped = NRE(previous_NRE, obs)
    previous_root = root_storage[f"round_{rd - 1}"]

    current_NRE = network_storage[f"round_{rd}"]
    current_NRE_wrapped = NRE(current_NRE, obs)
    current_root = root_storage[f"round_{rd}"]

    # load samples
    previous_samples = torch.load(f=f"{previous_root}/posterior_samples")
    current_samples = torch.load(f=f"{current_root}/posterior_samples")

    # get boundary samples
    previous_boundary_sample = torch.load(f=f"{previous_root}/boundary_sample")

    # evaluate new contour using previous boundary sample
    previous_boundary_logL = previous_NRE_wrapped.logLikelihood(previous_boundary_sample)

    if not nreSettings.ns_nre_use_previous_boundary_sample_for_counting:
        current_boundary_sample = torch.load(f=f"{current_root}/boundary_sample")
        current_boundary_logL = current_NRE_wrapped.logLikelihood(current_boundary_sample)
    else:
        current_boundary_logL = current_NRE_wrapped.logLikelihood(previous_boundary_sample)
    current_logLs_with_previous_samples = current_NRE_wrapped.logLikelihood(previous_samples)
    previous_logLs_with_current_samples = previous_NRE_wrapped.logLikelihood(current_samples)

    # count how many are within contour
    intersection_samples_A = current_logLs_with_previous_samples[
        current_logLs_with_previous_samples > current_boundary_logL]
    left_samples = current_logLs_with_previous_samples[current_logLs_with_previous_samples < current_boundary_logL]

    intersection_samples_B = previous_logLs_with_current_samples[
        previous_logLs_with_current_samples > previous_boundary_logL]
    right_samples = previous_logLs_with_current_samples[previous_logLs_with_current_samples < previous_boundary_logL]

    logger.info(f"# of previous samples within current NRE: {len(intersection_samples_A)}")
    logger.info(f"# of previous samples not within current NRE: {len(left_samples)}")
    logger.info(f"# of current samples within previous NRE: {len(intersection_samples_B)}")
    logger.info(f"# of current samples not within previous NRE: {len(right_samples)}")

    torch.save(obj=intersection_samples_A,
               f=f"{current_root}/previous_samples_within_current_NRE")  # save intersection samples
    torch.save(obj=intersection_samples_B,
               f=f"{current_root}/current_samples_within_previous_NRE")  # save intersection samples
    torch.save(obj=left_samples, f=f"{current_root}/previous_samples_not_within_current_NRE")  # save left samples
    torch.save(obj=right_samples, f=f"{current_root}/current_samples_not_within_previous_NRE")  # save right samples
    return intersection_samples_A, intersection_samples_B, left_samples, right_samples
