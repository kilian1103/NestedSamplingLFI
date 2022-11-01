from typing import Dict, Any

import matplotlib.pyplot as plt  #
import numpy as np
import scipy.special
import swyft
from anesthetic import NestedSamples

from NSLFI.MCMCSampler import Sampler
from NSLFI.NRE import NRE


def nested_sampling(ndim: int, nsim: int, stop_criterion: float, samplerType: str, trainedNRE: NRE,
                    x_0: Dict[str, np.ndarray]) -> Dict[str, Any]:
    # initialisation
    logZ_previous = -np.inf * np.ones(nsim)  # Z = 0
    logX_previous = np.zeros(nsim)  # X = 1
    iteration = 0
    logIncrease = 10  # evidence increase factor

    # fixed length storage -> nd.array
    livepoints = trainedNRE.dataset.v.copy()
    nlive = len(livepoints)
    logLikelihoods = trainedNRE.mre_2d.log_ratio(observation=x_0, v=livepoints)[trainedNRE.marginal_indices_2d].copy()
    livepoints_birthlogL = -np.inf * np.ones(nlive)  # L_birth = 0

    # dynamic storage -> lists
    deadpoints = []
    deadpoints_logL = []
    deadpoints_birthlogL = []
    weights = []

    sampler = Sampler(prior=trainedNRE.prior, priorLimits=trainedNRE.priorLimits,
                      logLikelihood=trainedNRE.logLikelihood,
                      ndim=ndim).getSampler(samplerType)
    while logIncrease > np.log(stop_criterion):
        iteration += 1
        # identifying lowest likelihood point
        minlogLike = logLikelihoods.min()
        index = logLikelihoods.argmin()

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
        proposal_sample = sampler.sample(livepoints=livepoints.copy(), minlogLike=minlogLike)

        # replace lowest likelihood sample with proposal sample
        livepoints[index] = proposal_sample.copy().tolist()
        logLikelihoods[index] = float(
            trainedNRE.mre_2d.log_ratio(observation=x_0, v=[proposal_sample])[trainedNRE.marginal_indices_2d].copy())
        livepoints_birthlogL[index] = minlogLike
        # add datapoint to NRE
        trainedNRE.store._append_new_points(v=[proposal_sample],
                                            log_w=trainedNRE.mre_2d.log_ratio(observation=x_0, v=[proposal_sample])[
                                                trainedNRE.marginal_indices_2d])

        maxlogLike = logLikelihoods.max()
        logIncrease_array = logWeight_current + maxlogLike - logZ_total
        # logIncrease_array = logWeight_current + maxlogRatio - logZ_total
        logIncrease = logIncrease_array.max()
        if iteration % 500 == 0:
            print("Current log evidence ", logZ_total.max())
            print("current iteration: ", iteration)

    # final <L>*dX sum calculation
    finallogLikesum = scipy.special.logsumexp(a=logLikelihoods)
    logZ_current = -np.log(nlive) + finallogLikesum + logX_current
    logZ_array = np.array([logZ_previous, logZ_current])
    logZ_total = scipy.special.logsumexp(logZ_array, axis=0)

    # convert surviving livepoints to deadpoints
    livepoints = livepoints.tolist()
    logLikelihoods = logLikelihoods.tolist()
    while len(logLikelihoods) > 0:
        minlogLike = min(logLikelihoods)
        index = logLikelihoods.index(minlogLike)

        deadpoint = livepoints.pop(index)
        logLikelihoods.pop(index)

        deadpoints.append(deadpoint)
        deadpoints_logL.append(minlogLike)
        deadpoints_birthlogL.append(livepoints_birthlogL[index])
        weights.append(np.mean(logX_current) - np.log(nlive))

    np.save(file="weights", arr=np.array(weights))
    np.save(file="posterior_samples", arr=np.array(deadpoints))
    np.save(file="logL", arr=np.array(deadpoints_logL))
    np.save(file="logL_birth", arr=np.array(deadpoints_birthlogL))
    nested = NestedSamples(data=deadpoints, weights=weights, logL_birth=np.array(deadpoints_birthlogL),
                           logL=np.array(deadpoints_logL))
    plt.figure()
    nested.plot_2d([0, 1])
    plt.suptitle("NRE NS enhanced samples")
    plt.savefig(fname="swyft_data/afterNS.pdf")
    print(f"Algorithm terminated after {iteration} iterations!")

    # update store state dict
    trainedNRE.store.log_lambdas.resize(len(trainedNRE.store.log_lambdas) + 1)
    pdf = swyft.PriorTruncator(trainedNRE.prior, bound=None)
    trainedNRE.store.log_lambdas[-1] = dict(pdf=pdf.state_dict(),
                                            N=trainedNRE.nre_settings.n_training_samples + iteration)
    # retrain NRE
    trainedNRE.store.simulate()
    trainedNRE.dataset = swyft.Dataset(trainedNRE.nre_settings.n_training_samples + iteration,
                                       trainedNRE.prior,
                                       trainedNRE.store)
    trainedNRE.store.save(path=trainedNRE.nre_settings.store_filename_NSenhanced)
    trainedNRE.dataset.save(trainedNRE.nre_settings.dataset_filename_NSenhanced)
    trainedNRE.mre_2d.train(trainedNRE.dataset)
    trainedNRE.mre_2d.save(trainedNRE.nre_settings.mre_2d_filename_NSenhanced)
    return {"log Z mean": np.mean(logZ_total),
            "log Z std": np.std(logZ_total),
            "retrainedNRE": trainedNRE}
