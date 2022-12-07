import numpy as np

from NSLFI.MCMCSampler import Sampler


def nested_sampling(logLikelihood, prior, livepoints, nsim, stop_criterion, samplertype, nSamples):
    # initialisation

    logLikelihoods = logLikelihood(livepoints)

    # dynamic storage -> lists
    deadpoints = []
    deadpoints_logL = []
    deadpoints_birthlogL = []

    sampler = Sampler(prior=prior, logLikelihood=logLikelihood).getSampler(samplertype)

    # identifying median likelihood point
    medianlogLike = np.median(logLikelihoods)

    iteration = nSamples
    for it in range(iteration):
        # find new sample satisfying likelihood constraint
        proposal_sample = sampler.sample(livepoints=livepoints.copy(), minlogLike=medianlogLike)
        while float(logLikelihood(proposal_sample)) < medianlogLike:
            proposal_sample = sampler.sample(livepoints=livepoints.copy(), minlogLike=medianlogLike)
        # add new sample to deadpoints
        deadpoints.append(proposal_sample)
        deadpoints_birthlogL.append(medianlogLike)
        deadpoints_logL.append(float(logLikelihood(proposal_sample)))

    np.save(file="posterior_samples_roundNS", arr=np.array(deadpoints))
    np.save(file="logL_roundNS", arr=np.array(deadpoints_logL))
    np.save(file="logL_birth_roundNS", arr=np.array(deadpoints_birthlogL))
    print(f"Algorithm terminated after {iteration} iterations!")
    return
