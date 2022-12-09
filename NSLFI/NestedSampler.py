import numpy as np
import scipy.special

from NSLFI.MCMCSampler import Sampler


def nested_sampling(logLikelihood, prior, livepoints, nsim, stop_criterion, samplertype, rounds=0, iter=2000,
                    root="."):
    if rounds == 0:
        # standard NS run
        # initialisation
        logZ_previous = -np.inf * np.ones(nsim)  # Z = 0
        logX_previous = np.zeros(nsim)  # X = 1
        iteration = 0
        logIncrease = 10  # evidence increase factor
        nlive = livepoints.shape[0]

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

            # find new sample satisfying likelihood constraint
            proposal_sample = sampler.sample(livepoints=livepoints.copy(), minlogLike=minlogLike)
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
        return {"log Z mean": np.mean(logZ_total),
                "log Z std": np.std(logZ_total)}

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
            for it in range(iter):
                # find new sample satisfying likelihood constraint
                proposal_sample = sampler.sample(livepoints=livepoints.copy(), minlogLike=medianlogLike)
                while float(logLikelihood(proposal_sample)) < medianlogLike:
                    proposal_sample = sampler.sample(livepoints=livepoints.copy(), minlogLike=medianlogLike)
                # add new sample to deadpoints
                deadpoints.append(proposal_sample)
                deadpoints_birthlogL.append(medianlogLike)
                deadpoints_logL.append(float(logLikelihood(proposal_sample)))
            np.save(file=f"{root}/posterior_samples_rounds_{rd}", arr=np.array(deadpoints))
            np.save(file=f"{root}/logL_rounds_{rd}", arr=np.array(deadpoints_logL))
            np.save(file=f"{root}/logL_birth_rounds_{rd}", arr=np.array(deadpoints_birthlogL))
            livepoints = np.array(deadpoints.copy())
        return {"log Z mean": 0, "log Z std": 0}
