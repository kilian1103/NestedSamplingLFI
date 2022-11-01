import logging

import numpy as np
import pypolychord
from margarine.maf import MAF
from mpi4py import MPI
from pypolychord import PolyChordSettings

from NSLFI.MAF_Polychord_Wrapper import MAF_Poly

logging.basicConfig(filename="myLFI.log", level=logging.INFO,
                    filemode="w")
logger = logging.getLogger()
logger.info('Started')
file = f"toyproblem_bij.pkl"
theta = np.load("posterior_samples.npy")
weights = np.load("weights.npy")

# train bijcettors
# bij = MAF(theta, weights)
# bij.train(epochs=500)
# bij.save(file)

# load bij
bij = MAF.load(file)
# wrap MAF for Polychord
poly_MAF = MAF_Poly(bij)
polychordSet = PolyChordSettings(nDims=poly_MAF.nDims, nDerived=poly_MAF.nDerived)
polychordSet.nlive = 100

try:
    comm_analyse = MPI.COMM_WORLD
    rank_analyse = comm_analyse.Get_rank()
except Exception as e:
    logger.error(
        "Oops! {} occurred. when Get_rank()".format(e.__class__))
    rank_analyse = 0


def dumper(live, dead, logweights, logZ, logZerr):
    """Dumper Function for PolyChord for runtime progress access."""
    logger.info("Last dead point: {}".format(dead[-1]))


# compute integral of MAF
output = pypolychord.run_polychord(poly_MAF.loglike,
                                   poly_MAF.nDims,
                                   poly_MAF.nDerived,
                                   polychordSet,
                                   poly_MAF.prior, dumper)
comm_analyse.Barrier()
logger.info("Done")
