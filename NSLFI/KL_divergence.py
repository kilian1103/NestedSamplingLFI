from typing import Dict
from typing import Tuple

import anesthetic
import torch

from NSLFI.NRE_Polychord_Wrapper import NRE_PolyChord
from NSLFI.NRE_Settings import NRE_Settings


def compute_KL_divergence(nreSettings: NRE_Settings, network_storage: Dict[str, NRE_PolyChord],
                          current_samples: anesthetic.Samples, rd: int) -> Tuple[float, float]:
    """Compute the KL divergence between the previous and current NRE."""
    previous_network = network_storage[f"round_{rd - 1}"]
    with torch.no_grad():
        current_samples["logL_previous"] = \
            previous_network.logLikelihood(current_samples.iloc[:, :nreSettings.num_features].to_numpy())[0]
    DKL = (current_samples["logL"] - current_samples["logL_previous"]).mean()
    DKL_err = (current_samples["logL"] - current_samples["logL_previous"]).std()
    return DKL, DKL_err
