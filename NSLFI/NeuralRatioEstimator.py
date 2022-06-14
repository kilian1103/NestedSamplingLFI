import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import swyft
from anesthetic import MCMCSamples

np.random.seed(234)
### swyft mode###
# mode = "train"
mode = "load"
MNREmode = False
device = "cpu"
n_training_samples = 10_000
n_weighted_samples = 10_000

# true parameters of simulator
theta_0 = np.array([5, 25, 10])
paramNames = [r"$\sigma$", r"$f_0$", r"$A$"]

# saving file names
prior_filename = "swyft_data/toyproblem.prior.pt"
dataset_filename = "swyft_data/toyproblem.dataset.pt"
mre_1d_filename = "swyft_data/toyproblem.mre_1d.pt"
mre_2d_filename = "swyft_data/toyproblem.mre_2d.pt"
mre_3d_filename = "swyft_data/toyproblem.mre_3d.pt"
store_filename = "swyft_data/SavedStore"


# simulator
def forwardmodel(theta):
    freq = np.arange(0, 50, 0.5)
    sigma = theta[0]
    f0 = theta[1]
    A = theta[2]
    x = A * np.exp(-0.5 * (freq - f0) ** 2 / sigma ** 2)
    # x = np.random.normal(loc=x, scale = 0.5)
    return {"x": x}


# create real observation
x_0 = forwardmodel(theta_0)
x_0["x"] = np.random.normal(loc=x_0["x"], scale=0.5)
freq = np.arange(0, 50, 0.5)

# plot observation
plt.figure()
plt.title(r"Observation $x_0$ to fit")
plt.xlabel("Frequency")
plt.ylabel("Signal strength")
plt.plot(freq, x_0["x"])
plt.show()

# initialize swyft
n_parameters = len(theta_0)
observation_key = "x"
observation_shapes = {observation_key: x_0[observation_key].shape}
simulator = swyft.Simulator(
    forwardmodel,
    n_parameters,
    sim_shapes=observation_shapes
)
# assign prior for each 1-dim parameter
uniform_scipy_1 = scipy.stats.uniform(0, 10)
uniform_scipy_2 = scipy.stats.uniform(15, 35)
uniform_scipy_3 = scipy.stats.uniform(0, 20)
prior = swyft.prior.Prior.composite_prior(
    cdfs=[uniform_scipy_1.cdf, uniform_scipy_2.cdf, uniform_scipy_3.cdf],
    icdfs=[uniform_scipy_1.ppf, uniform_scipy_2.ppf, uniform_scipy_3.ppf],
    log_probs=[uniform_scipy_1.logpdf, uniform_scipy_2.logpdf, uniform_scipy_3.logpdf],
    parameter_dimensions=[1 for x in range(n_parameters)],
)
store = swyft.Store.memory_store(simulator)

# get marginal indices (here fit 1d,2d and 3d marginals for 3d problem)
marginal_indices_1d, marginal_indices_2d = swyft.utils.get_corner_marginal_indices(n_parameters)
marginal_indices_3d = tuple([x for x in range(len(theta_0))])

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

network_3d = swyft.get_marginal_classifier(
    observation_key=observation_key,
    marginal_indices=marginal_indices_3d,
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

    mre_3d = swyft.MarginalRatioEstimator(
        marginal_indices=marginal_indices_3d,
        network=network_3d,
        device=device,
    )
    mre_3d.train(dataset)
    mre_3d.save(mre_3d_filename)
# load MRE from file
else:
    store = swyft.Store.load(store_filename, simulator=simulator).to_memory()
    prior = swyft.Prior.load(prior_filename)
    dataset = swyft.Dataset.load(
        filename=dataset_filename,
        store=store
    )
    if MNREmode:
        mre_1d = swyft.MarginalRatioEstimator.load(
            network=network_1d,
            device=device,
            filename=mre_1d_filename,
        )

        mre_2d = swyft.MarginalRatioEstimator.load(
            network=network_2d,
            device=device,
            filename=mre_2d_filename,
        )

    mre_3d = swyft.MarginalRatioEstimator.load(
        network=network_3d,
        device=device,
        filename=mre_3d_filename,
    )

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

posterior_3d = swyft.MarginalPosterior(mre_3d, prior)
weighted_samples_3d = posterior_3d.weighted_sample(n_weighted_samples, x_0)
data = weighted_samples_3d.get_df(marginal_indices_3d)

columnNames = {}
for i, j in enumerate(paramNames):
    columnNames[i] = j
data.rename(columns=columnNames, inplace=True)
mcmc = MCMCSamples(data=data, weights=data.weight)
plt.figure()
mcmc.plot_2d(axes=paramNames)
plt.suptitle("NRE parameter estimations")
plt.show()
logProb_0 = posterior_3d.log_prob(observation=x_0, v=[theta_0])
print(f"log probability of theta_0 using NRE is: {float(logProb_0[marginal_indices_3d]):.3f}")