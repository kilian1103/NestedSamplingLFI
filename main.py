import logging

import matplotlib.pyplot as plt
import numpy as np
import pypolychord
import scipy.special
import swyft
from anesthetic import MCMCSamples, NestedSamples
from mpi4py import MPI
from pypolychord import PolyChordSettings
from scipy.stats import multivariate_normal

import NSLFI.NestedSampler
from NSLFI.NRE import NRE
from NSLFI.NRE_PolyChord_Wrapper import NRE_Poly
from NSLFI.NRE_Settings import NRE_Settings

np.random.seed(234)
logging.basicConfig(filename="myLFI.log", level=logging.INFO,
                    filemode="w")
logger = logging.getLogger()
logger.info('Started')
nreSettings = NRE_Settings()
nreSettings.n_training_samples = 1000
nreSettings.n_weighted_samples = 10000
nreSettings.mode = "load"
priorLimits = {"lower": np.array([0, 0]),
               "upper": np.array([1, 1])}
### swyft mode###
mode = "train"
# mode = nreSettings.mode
MNREmode = nreSettings.MNREmode
simulatedObservations = nreSettings.simulatedObservations
simulatedObservations = True
device = nreSettings.device
n_training_samples = nreSettings.n_training_samples
n_weighted_samples = nreSettings.n_weighted_samples

# true parameters of simulator
theta_0 = nreSettings.theta_0
paramNames = nreSettings.paramNames
n_parameters = nreSettings.n_parameters

# saving file names
prior_filename = nreSettings.prior_filename
dataset_filename = nreSettings.dataset_filename
mre_2d_filename = nreSettings.mre_2d_filename
store_filename = nreSettings.store_filename
observation_filename = nreSettings.observation_filename

# simulator
observation_key = nreSettings.observation_key


def forwardmodel(theta, ndim=2):
    # data space
    nData = 10
    # estimate 2d mean from multivariate normal distribution
    means = np.array([theta[0], theta[1]]) * np.ones(shape=ndim)
    cov = 0.01 * np.eye(N=ndim)
    x = multivariate_normal.rvs(mean=means, cov=cov, size=nData).reshape(ndim, nData)
    return {observation_key: x}


# create (simulated real) observation data
if simulatedObservations:
    x_0 = forwardmodel(theta_0)
    np.save(file=observation_filename, arr=x_0[observation_key])
else:
    # else load from file
    x_0 = np.load(file=observation_filename, allow_pickle=True)
    x_0 = {observation_key: x_0}

# plot observation
plt.figure()
plt.title(r"Observation $x_0$ to fit")
plt.xlabel("X_1")
plt.ylabel("X_2")
plt.scatter(x_0[observation_key][0], x_0[observation_key][1])
plt.savefig(fname="swyft_data/observation.pdf")

# initialize swyft
observation_shapes = {observation_key: x_0[observation_key].shape}
simulator = swyft.Simulator(
    forwardmodel,
    n_parameters,
    sim_shapes=observation_shapes
)

# assign prior for each 1-dim parameter
uniform_scipy_1 = scipy.stats.uniform(loc=priorLimits["lower"][0],
                                      scale=priorLimits["upper"][0] - priorLimits["lower"][0])
uniform_scipy_2 = scipy.stats.uniform(loc=priorLimits["lower"][1],
                                      scale=priorLimits["upper"][1] - priorLimits["lower"][1])
prior = swyft.prior.Prior.composite_prior(
    cdfs=[uniform_scipy_1.cdf, uniform_scipy_2.cdf],
    icdfs=[uniform_scipy_1.ppf, uniform_scipy_2.ppf],
    log_probs=[uniform_scipy_1.logpdf, uniform_scipy_2.logpdf],
    parameter_dimensions=[1 for x in range(n_parameters)],
)
store = swyft.Store.memory_store(simulator)

# get marginal indices (here fit 1d,2d and 3d marginals for 3d problem)
marginal_indices_2d = (0, 1)

# define networks
network_2d = swyft.get_marginal_classifier(
    observation_key=observation_key,
    marginal_indices=marginal_indices_2d,
    observation_shapes=observation_shapes,
    n_parameters=n_parameters,
    hidden_features=32,
    num_blocks=2,
)

# train MRE
if mode == "train":
    # initialize simulator and simulate
    store.add(n_training_samples, prior)
    store.simulate()
    dataset = swyft.Dataset(n_training_samples, prior, store)
    # save objects before training network
    prior.save(prior_filename)
    store.save(store_filename)
    dataset.save(dataset_filename)

    mre_2d = swyft.MarginalRatioEstimator(
        marginal_indices=marginal_indices_2d,
        network=network_2d,
        device=device,
    )
    mre_2d.train(dataset)
    mre_2d.save(mre_2d_filename)

# load MRE from file
else:
    store = swyft.Store.load(store_filename, simulator=simulator).to_memory()
    prior = swyft.Prior.load(prior_filename)
    dataset = swyft.Dataset.load(
        filename=dataset_filename,
        store=store
    )

    mre_2d = swyft.MarginalRatioEstimator.load(
        network=network_2d,
        device=device,
        filename=mre_2d_filename,
    )

# get posterior samples
posterior = swyft.MarginalPosterior(mre_2d, prior)
weighted_samples_2d = posterior.weighted_sample(n_weighted_samples, x_0)
data = weighted_samples_2d.get_df(marginal_indices_2d)

# plot initial NRE
columnNames = {}
for i, j in enumerate(paramNames):
    columnNames[i] = j
data.rename(columns=columnNames, inplace=True)
mcmc = MCMCSamples(data=data, weights=data.weight)
plt.figure()
mcmc.plot_2d(axes=paramNames)
plt.suptitle("NRE parameter estimation")
plt.savefig(fname="swyft_data/firstNRE.pdf")
logProb_0 = posterior.log_prob(observation=x_0, v=[theta_0])
logger.info(f"log probability of theta_0 using NRE is: {float(logProb_0[marginal_indices_2d]):.3f}")

# wrap NRE object
trained_NRE = NRE(dataset=dataset, store=store, prior=prior, priorLimits=priorLimits, trained_NRE=mre_2d,
                  nreSettings=nreSettings, x_0=x_0)

# wrap NRE for Polychord
poly_NRE = NRE_Poly(nre=trained_NRE.mre_2d, x_0=x_0)
polychordSet = PolyChordSettings(nDims=poly_NRE.nDims, nDerived=poly_NRE.nDerived)
polychordSet.nlive = n_training_samples
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


output = pypolychord.run_polychord(poly_NRE.loglike,
                                   poly_NRE.nDims,
                                   poly_NRE.nDerived,
                                   polychordSet,
                                   poly_NRE.prior, dumper)
comm_analyse.Barrier()

# optimize with my NS run
output = NSLFI.NestedSampler.nested_sampling(logLikelihood=trained_NRE.logLikelihood,
                                             livepoints=trained_NRE.dataset.v.copy(), prior=trained_NRE.prior,
                                             priorLimits=trained_NRE.priorLimits, ndim=trained_NRE.ndim, nsim=100,
                                             stop_criterion=1e-3,
                                             samplertype="Metropolis")
logger.info(output)

deadpoints = np.load(file="posterior_samples.npy")
weights = np.load(file="weights.npy")
deadpoints_birthlogL = np.load(file="logL_birth.npy")
deadpoints_logL = np.load(file="logL.npy")
newPoints = np.load(file="newPoints.npy")
nested = NestedSamples(data=deadpoints, weights=weights, logL_birth=deadpoints_birthlogL,
                       logL=deadpoints_logL)
plt.figure()
nested.plot_2d([0, 1])
plt.suptitle("NRE NS enhanced samples")
plt.savefig(fname="swyft_data/afterNS.pdf")

# add datapoint to NRE
iteration = 0
logger.info(f"Adding newly found datapoints to NRE")
for proposal_sample in newPoints:
    iteration += 1
    trained_NRE.store._append_new_points(v=[proposal_sample],
                                         log_w=trained_NRE.mre_2d.log_ratio(observation=x_0, v=[proposal_sample])[
                                             trained_NRE.marginal_indices_2d])
logger.info(f"Added {iteration} newly found datapoints to NRE")
# update store state dict
trained_NRE.store.log_lambdas.resize(len(trained_NRE.store.log_lambdas) + 1)
pdf = swyft.PriorTruncator(trained_NRE.prior, bound=None)
trained_NRE.store.log_lambdas[-1] = dict(pdf=pdf.state_dict(),
                                         N=trained_NRE.nre_settings.n_training_samples + iteration)
# retrain NRE
trained_NRE.store.simulate()
trained_NRE.dataset = swyft.Dataset(trained_NRE.nre_settings.n_training_samples + iteration,
                                    trained_NRE.prior,
                                    trained_NRE.store)
trained_NRE.store.save(path=trained_NRE.nre_settings.store_filename_NSenhanced)
trained_NRE.dataset.save(trained_NRE.nre_settings.dataset_filename_NSenhanced)
trained_NRE.mre_2d.train(trained_NRE.dataset)
trained_NRE.mre_2d.save(trained_NRE.nre_settings.mre_2d_filename_NSenhanced)

weighted_samples_3d = posterior.weighted_sample(trained_NRE.nre_settings.n_weighted_samples * 10, x_0)
data = weighted_samples_3d.get_df(trained_NRE.marginal_indices_2d)
columnNames = {}
for i, j in enumerate(trained_NRE.nre_settings.paramNames):
    columnNames[i] = j
data.rename(columns=columnNames, inplace=True)
mcmc = MCMCSamples(data=data, weights=data.weight)
plt.figure()
mcmc.plot_2d(axes=trained_NRE.nre_settings.paramNames)
plt.suptitle("Retrained NRE parameter estimations")
plt.savefig(fname="swyft_data/retrained_NRE.pdf")

logger.info("Done")
