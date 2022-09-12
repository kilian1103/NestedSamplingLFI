import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import swyft
from anesthetic import MCMCSamples
from scipy.stats import multivariate_normal

import NSLFI.NSNRE
from NSLFI.NRE import NRE
from NSLFI.NRE_Settings import NRE_Settings

np.random.seed(234)

nreSettings = NRE_Settings()
nreSettings.n_training_samples = 1000
nreSettings.n_weighted_samples = 10000
nreSettings.mode = "load"
priorLimits = {"lower": np.array([0, 0]),
               "upper": np.array([1, 1])}
### swyft mode###
# mode = "train"
mode = nreSettings.mode
MNREmode = nreSettings.MNREmode
simulatedObservations = nreSettings.simulatedObservations
# simulatedObservations = True
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
mre_1d_filename = nreSettings.mre_1d_filename
mre_2d_filename = nreSettings.mre_2d_filename
store_filename = nreSettings.store_filename
observation_filename = nreSettings.observation_filename

# simulator
observation_key = nreSettings.observation_key


def forwardmodel(theta, ndim=2):
    # data space
    nData = 100
    # estimate mean and cov from multivariate normal distribution
    means = theta[0] * np.ones(shape=ndim)
    cov = theta[1] * np.eye(N=ndim)
    x = multivariate_normal.rvs(mean=means, cov=cov, size=(nData)).reshape(ndim, nData)
    return {observation_key: x}


# create (simulated real) observation data
if simulatedObservations:
    x_0 = forwardmodel(theta_0)
    # x_0[observation_key] = np.random.normal(loc=x_0[observation_key], scale=0.5)
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
marginal_indices_1d, marginal_indices_2d = swyft.utils.get_corner_marginal_indices(n_parameters)
# marginal_indices_3d = tuple([x for x in range(len(theta_0))])
marginal_indices_2d = (0, 1)
# define networks
network_1d = swyft.get_marginal_classifier(
    observation_key=observation_key,
    marginal_indices=marginal_indices_1d,
    observation_shapes=observation_shapes,
    n_parameters=n_parameters,
    hidden_features=32,
    num_blocks=2,
)
network_2d = swyft.get_marginal_classifier(
    observation_key=observation_key,
    marginal_indices=marginal_indices_2d,
    observation_shapes=observation_shapes,
    n_parameters=n_parameters,
    hidden_features=32,
    num_blocks=2,
)
#
# network_3d = swyft.get_marginal_classifier(
#     observation_key=observation_key,
#     marginal_indices=marginal_indices_3d,
#     observation_shapes=observation_shapes,
#     n_parameters=n_parameters,
#     hidden_features=32,
#     num_blocks=2,
# )

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
    if MNREmode:
        mre_1d = swyft.MarginalRatioEstimator(
            marginal_indices=marginal_indices_1d,
            network=network_1d,
            device=device,
        )
        mre_1d.train(dataset)
        mre_1d.save(mre_1d_filename)

    mre_2d = swyft.MarginalRatioEstimator(
        marginal_indices=marginal_indices_2d,
        network=network_2d,
        device=device,
    )
    mre_2d.train(dataset)
    mre_2d.save(mre_2d_filename)

    # mre_3d = swyft.MarginalRatioEstimator(
    #     marginal_indices=marginal_indices_3d,
    #     network=network_3d,
    #     device=device,
    # )
    # mre_3d.train(dataset)
    # mre_3d.save(mre_3d_filename)
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

    # mre_3d = swyft.MarginalRatioEstimator.load(
    #     network=network_3d,
    #     device=device,
    #     filename=mre_3d_filename,
    # )

# get posterior samples
if MNREmode:
    posterior_1d = swyft.MarginalPosterior(mre_1d, prior)
    weighted_samples_1d = posterior_1d.weighted_sample(n_weighted_samples, x_0)
    posterior_2d = swyft.MarginalPosterior(mre_2d, prior)
    weighted_samples_2d = posterior_2d.weighted_sample(n_weighted_samples, x_0)
    plt.figure()
    _, _ = swyft.corner(
        weighted_samples_1d,
        weighted_samples_2d,
        kde=True,
        truth=theta_0,
        labels=paramNames
    )
    plt.suptitle("MNRE parameter estimation")
    plt.show()

posterior = swyft.MarginalPosterior(mre_2d, prior)
weighted_samples_2d = posterior.weighted_sample(n_weighted_samples, x_0)
data = weighted_samples_2d.get_df(marginal_indices_2d)

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
print(f"log probability of theta_0 using NRE is: {float(logProb_0[marginal_indices_2d]):.3f}")

trained_NRE = NRE(dataset=dataset, store=store, prior=prior, priorLimits=priorLimits, trainedNRE=mre_2d,
                  nreSettings=nreSettings)

output = NSLFI.NSNRE.nested_sampling(ndim=2, nsim=100, stop_criterion=1e-3,
                                     samplerType="MetropolisNRE", trainedNRE=trained_NRE, x_0=x_0)

trained_nre = output["retrainedNRE"]
weighted_samples_3d = posterior.weighted_sample(trained_nre.nre_settings.n_weighted_samples * 10, x_0)
data = weighted_samples_3d.get_df(trained_nre.marginal_indices_2d)
columnNames = {}
for i, j in enumerate(trained_nre.nre_settings.paramNames):
    columnNames[i] = j
data.rename(columns=columnNames, inplace=True)
mcmc = MCMCSamples(data=data, weights=data.weight)
plt.figure()
mcmc.plot_2d(axes=trained_nre.nre_settings.paramNames)
plt.suptitle("Retrained NRE parameter estimations")
plt.savefig(fname="swyft_data/retrained_NRE.pdf")
