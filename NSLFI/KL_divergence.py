from typing import Dict

import anesthetic
import torch

from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings


def compute_KL_divergence(nreSettings: NRE_Settings, network_storage: Dict[str, NRE_PolyChord],
                          current_samples: anesthetic.NestedSamples, rd: int) -> float:
    """Compute the KL divergence between the previous and current NRE."""
    previous_network = network_storage[f"round_{rd - 1}"]
    with torch.no_grad():
        current_samples["logL_previous"] = \
            previous_network.logLikelihood(current_samples.iloc[:, :nreSettings.num_features].to_numpy())[0]
    normalised_weights = current_samples.get_weights() / current_samples.get_weights().sum()
    DKL = (normalised_weights * (
            current_samples["logL"] - current_samples["logL_previous"])).sum()
    return DKL
