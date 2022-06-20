import matplotlib.pyplot as plt  #
import numpy as np
import scipy.special
import swyft
from anesthetic import MCMCSamples
from scipy.stats import multivariate_normal

from NSLFI.MCMCSampler import Sampler
from NSLFI.NRE_Settings import NRE_Settings


def nested_sampling(ndim, nsim, stop_criterion, samplertype, nreSettings, totalIteration):
    # initialisation
    np.random.seed(234)
    logZ_previous = -np.inf * np.ones(nsim)  # Z = 0
    logX_previous = np.zeros(nsim)  # X = 1
    iteration = 0
    logIncrease = 10  # evidence increase factor

    ### swyft mode###
    # mode = "train"
    mode = nreSettings.mode
    MNREmode = nreSettings.MNREmode
    simulatedObservations = nreSettings.simulatedObservations
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
    mre_3d_filename = nreSettings.mre_3d_filename
    store_filename = nreSettings.store_filename
    observation_filename = nreSettings.observation_filename

    # simulator
    observation_key = nreSettings.observation_key

    def forwardmodel(theta):
        freq = np.arange(0, 50, 0.5)
        sigma = theta[0]
        f0 = theta[1]
        A = theta[2]
        x = A * np.exp(-0.5 * (freq - f0) ** 2 / sigma ** 2)
        # x = np.random.normal(loc=x, scale = 0.5)
        return {observation_key: x}

    # create (simulated real) observation data
    if simulatedObservations:
        x_0 = forwardmodel(theta_0)
        x_0[observation_key] = np.random.normal(loc=x_0[observation_key], scale=0.5)
        np.save(file=observation_filename, arr=x_0[observation_key])
    else:
        # else load from file
        x_0 = np.load(file=observation_filename, allow_pickle=True)
        x_0 = {observation_key: x_0}
    freq = np.arange(0, 50, 0.5)

    # plot observation
    plt.figure()
    plt.title(r"Observation $x_0$ to fit")
    plt.xlabel("Frequency")
    plt.ylabel("Signal strength")
    plt.plot(freq, x_0[observation_key])
    plt.show()

    # initialize swyft
    observation_shapes = {observation_key: x_0[observation_key].shape}
    simulator = swyft.Simulator(
        forwardmodel,
        n_parameters,
        sim_shapes=observation_shapes
    )
    # assign prior for each 1-dim parameter
    priorLimits = {"lower": [0, 15, 0],
                   "upper": [10, 35, 20]}
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

    # fixed length storage -> nd.array
    livepoints = dataset.v.copy()
    nlive = len(livepoints)
    logPosteriors = posterior_3d.log_prob(observation=x_0, v=livepoints)[marginal_indices_3d].copy()
    logRatios = mre_3d.log_ratio(observation=x_0, v=livepoints)[marginal_indices_3d].copy()
    livepoints_birthlogL = -np.inf * np.ones(nlive)  # L_birth = 0

    # dynamic storage -> lists
    deadpoints = []
    deadpoints_logL = []
    deadpoints_birthlogL = []
    weights = []

    sampler = Sampler(prior=prior, priorLimits=priorLimits, logLikelihood=posterior_3d, ndim=ndim).getSampler(
        samplertype)
    while logIncrease > np.log(stop_criterion):
        # for it in range(totalIteration):
        iteration += 1
        # identifying lowest likelihood point
        minlogLike = logPosteriors.min()
        index = logPosteriors.argmin()

        # save deadpoint and its loglike
        deadpoint = livepoints[index].copy()
        deadpoints.append(deadpoint)
        deadpoints_logL.append(minlogLike)
        deadpoints_birthlogL.append(livepoints_birthlogL[index].copy())

        # sample t's
        ti_s = np.random.power(a=nlive, size=nsim)
        log_ti_s = np.log(ti_s)

        # Calculate X contraction and weight
        logX_current = logX_previous + log_ti_s
        subtraction_coeff = np.array([1, -1]).reshape(2, 1)
        logWeights = np.array([logX_previous, logX_current])
        logWeight_current = scipy.special.logsumexp(a=logWeights, b=subtraction_coeff, axis=0)
        logX_previous = logX_current.copy()
        weights.append(np.mean(logWeight_current))

        # Calculate evidence increase
        logZ_current = logWeight_current + minlogLike
        logZ_array = np.array([logZ_previous, logZ_current])
        logZ_total = scipy.special.logsumexp(logZ_array, axis=0)
        logZ_previous = logZ_total.copy()

        # find new sample satisfying likelihood constraint
        proposal_sample = sampler.sample(livepoints=livepoints.copy(), minlogLike=minlogLike,
                                         marginal_indices_3d=marginal_indices_3d, x_0=x_0)

        # replace lowest likelihood sample with proposal sample
        livepoints[index] = proposal_sample.copy().tolist()
        logPosteriors[index] = float(posterior_3d.log_prob(observation=x_0, v=[proposal_sample])[
                                         marginal_indices_3d].copy())
        logRatios[index] = float(mre_3d.log_ratio(observation=x_0, v=[proposal_sample])[marginal_indices_3d].copy())
        livepoints_birthlogL[index] = minlogLike
        # add datapoint to NRE
        store._append_new_points(v=[proposal_sample],
                                 log_w=mre_3d.log_ratio(observation=x_0, v=[proposal_sample])[marginal_indices_3d])

        maxlogLike = logPosteriors.max()
        maxlogRatio = logRatios.max()
        logIncrease_array = logWeight_current + maxlogLike - logZ_total
        # logIncrease_array = logWeight_current + maxlogRatio - logZ_total
        logIncrease = logIncrease_array.max()
        if iteration % 500 == 0:
            print("current logIncrease ", logIncrease)
            print("current iteration: ", iteration)

    store.get_simulation_status()
    # final <L>*dX sum calculation
    finallogLikesum = scipy.special.logsumexp(a=logPosteriors)
    logZ_current = -np.log(nlive) + finallogLikesum + logX_current
    logZ_array = np.array([logZ_previous, logZ_current])
    logZ_total = scipy.special.logsumexp(logZ_array, axis=0)

    # convert surviving livepoints to deadpoints
    livepoints = livepoints.tolist()
    logPosteriors = logPosteriors.tolist()
    while len(logPosteriors) > 0:
        minlogLike = min(logPosteriors)
        index = logPosteriors.index(minlogLike)

        deadpoint = livepoints.pop(index)
        logPosteriors.pop(index)

        deadpoints.append(deadpoint)
        deadpoints_logL.append(minlogLike)
        deadpoints_birthlogL.append(livepoints_birthlogL[index])
        weights.append(np.mean(logX_current) - np.log(nlive))

    np.save(file="weights", arr=np.array(weights))
    np.save(file="posterior_samples", arr=np.array(deadpoints))
    np.save(file="logL", arr=np.array(deadpoints_logL))
    np.save(file="logL_birth", arr=np.array(deadpoints_birthlogL))
    print(f"Algorithm terminated after {iteration} iterations!")
    return {"log Z mean": np.mean(logZ_total),
            "log Z std": np.std(logZ_total)}


nre_settings = NRE_Settings()
logZ = nested_sampling(ndim=3, nsim=100, stop_criterion=1e-3,
                       samplertype="MetropolisNRE", nreSettings=nre_settings, totalIteration=10000)
print(logZ)
