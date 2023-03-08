from typing import List, Any, Dict

import numpy as np
import scipy.special

from NSLFI.MCMCSampler import Sampler


def nested_sampling(logLikelihood: Any, prior: Dict[str, Any], livepoints: List[np.ndarray], nsim: int,
                    stop_criterion: float, samplertype: str, rounds=0, nsamples=2000,
                    root=".", keep_chain=False) -> Dict[str, float]:
    """
    :param logLikelihood: loglikelihood function given parameters for obs x
    :param prior: uniform prior distribution for parameters
    :param livepoints: list of livepoints
    :param nsim: number of parallel NS contractions
    :param stop_criterion: evidence stopping criterion
    :param samplertype: MCMC sampler type to draw new proposal samples
    :param rounds: # of rounds of NS run, 0 = standard NS run
    :param nsamples: number of samples to draw per round
    :param root: root file directory to store results
    :param keep_chain: keep intermediate MCMC chain of samples
    :return:
    """
    if rounds == 0:
        # standard NS run
        # initialisation
        logZ_previous = -np.inf * np.ones(nsim)  # Z = 0
        logX_previous = np.zeros(nsim)  # X = 1
        iteration = 0
        logIncrease = 10  # evidence increase factor
        nlive = livepoints.shape[0]
        cov = np.cov(livepoints.T)
        cholesky = np.linalg.cholesky(cov)

        logLikelihoods = logLikelihood(livepoints)
        livepoints_birthlogL = -np.inf * np.ones(nlive)  # L_birth = 0

        # dynamic storage -> lists
        deadpoints = []
        deadpoints_logL = []
        deadpoints_birthlogL = []
        weights = []
        newPoints = []

        sampler = Sampler(prior=prior, logLikelihood=logLikelihood).getSampler(
            samplertype)
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

            # recompute cov of livepoints
            if iteration % nlive == 0:
                cov = np.cov(livepoints.T)
                cholesky = np.linalg.cholesky(cov)
            # find new sample satisfying likelihood constraint
            proposal_samples = sampler.sample(livepoints=livepoints.copy(), minlogLike=minlogLike,
                                              livelikes=logLikelihoods, cov=cov, cholesky=cholesky,
                                              keep_chain=keep_chain)
            proposal_sample = proposal_samples.pop()
            newPoints.append(proposal_sample)

            # replace lowest likelihood sample with proposal sample
            livepoints[index] = proposal_sample.copy().tolist()
            logLikelihoods[index] = float(logLikelihood(proposal_sample))
            livepoints_birthlogL[index] = minlogLike

            maxlogLike = logLikelihoods.max()
            logIncrease_array = logWeight_current + maxlogLike - logZ_total
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
        np.save(file=f"{root}/weights", arr=np.array(weights))
        np.save(file=f"{root}/posterior_samples", arr=np.array(deadpoints))
        np.save(file=f"{root}/logL", arr=np.array(deadpoints_logL))
        np.save(file=f"{root}/logL_birth", arr=np.array(deadpoints_birthlogL))
        np.save(file=f"{root}/newPoints", arr=np.array(newPoints))
        print(f"Algorithm terminated after {iteration} iterations!")
        return {"log Z mean": float(np.mean(logZ_total)),
                "log Z std": float(np.std(logZ_total))}

    else:
        # NS run with rounds, and constant median Likelihood constraint for each round
        for rd in range(rounds):
            logLikelihoods = logLikelihood(livepoints)

            # dynamic storage -> lists
            deadpoints = []
            deadpoints_logL = []
            deadpoints_birthlogL = []

            sampler = Sampler(prior=prior, logLikelihood=logLikelihood).getSampler(samplertype)

            # find new sample satisfying likelihood constraint
            medianlogLike = np.median(logLikelihoods)
            livepoints = livepoints[logLikelihoods > medianlogLike]
            logLikelihoods = logLikelihoods[logLikelihoods > medianlogLike]
            cov = np.cov(livepoints.T)
            cholesky = np.linalg.cholesky(cov)

            while len(deadpoints) <= nsamples:
                # find new sample satisfying likelihood constraint
                proposal_samples = sampler.sample(livepoints=livepoints.copy(), minlogLike=medianlogLike,
                                                  livelikes=logLikelihoods, cov=cov, cholesky=cholesky,
                                                  keep_chain=keep_chain)
                while len(proposal_samples) > 0:
                    proposal_sample = proposal_samples.pop()
                    deadpoints.append(proposal_sample)
                    deadpoints_birthlogL.append(medianlogLike)
                    deadpoints_logL.append(float(logLikelihood(proposal_sample)))
                # add new sample to deadpoints
            np.save(file=f"{root}/posterior_samples_rounds_{rd}", arr=np.array(deadpoints))
            np.save(file=f"{root}/logL_rounds_{rd}", arr=np.array(deadpoints_logL))
            np.save(file=f"{root}/logL_birth_rounds_{rd}", arr=np.array(deadpoints_birthlogL))
            livepoints = np.array(deadpoints.copy())
        return {"log Z mean": 0, "log Z std": 0}
