import numpy as np
import matplotlib.pyplot as plt
import scipy.special
def logLikelihood(x, C, sigma=0.2):
    return -x**(2/C)/(2*sigma**2)

def prior(C, n_samples):
    #random_directions = np.random.normal(silogze=(C,n_samples))
    #norm = np.linalg.norm(random_directions, axis=0)
    #random_directions/=norm
    radii = np.random.random(n_samples)**(1/C)
    return radii**2

def nested_sampling(logLikelihood, prior, n_dim, nlive, stop_criterion):
    #initialisation
    logZ_previous = -1e300 # Z = 0
    logX_previous = 0 # X = 1
    iteration = 0
    logIncrease = 10 # evidence increase factor

    #sample from prior
    samples = prior(n_dim, nlive)
    logLikelihoods = logLikelihood(samples, n_dim)
    samples = samples.tolist()
    logLikelihoods = logLikelihoods.tolist()


    while logIncrease > np.log(stop_criterion):
        iteration += 1
        minlogLike = min(logLikelihoods)
        index = logLikelihoods.index(minlogLike)

        logX_current = -iteration/nlive

        subtraction_coeff = np.array([1,-1])
        logWeights = np.array([logX_previous, logX_current])
        logWeight_current =scipy.special.logsumexp(a=logWeights,b=subtraction_coeff)
        logX_previous = logX_current

        #TODO Fix log sum exp logic
        logZ_current = logWeight_current + minlogLike
        logZ_array = np.array([logZ_previous, logZ_current])
        logZ_total =  scipy.special.logsumexp(logZ_array)
        logZ_previous = logZ_total


        sampling = True
        while sampling:
            proposal_sample = prior(n_dim,1)

            if logLikelihood(proposal_sample,n_dim) > minlogLike:
                #accept
                samples[index] = float(proposal_sample)
                logLikelihoods[index] = float(logLikelihood(proposal_sample,n_dim))
                sampling = False

        maxlogLike = max(logLikelihoods)
        logIncrease = logWeight_current+maxlogLike-logZ_total
        if iteration%1000 == 0:
            print("current iteration: ", iteration)
            #print("current increase: ", increase)

    finallogLikelihoods = logLikelihood(np.array(samples), n_dim)
    #TODO Fix log Z addition
    finallogLikesum = scipy.special.logsumexp(a=finallogLikelihoods)
    logZ_current = -np.log(nlive) + finallogLikesum + logX_current
    logZ_array = np.array([logZ_previous, logZ_current])
    logZ_total = scipy.special.logsumexp(logZ_array)


    return logZ_total


logZ =nested_sampling(logLikelihood=logLikelihood, prior=prior, n_dim=2,nlive=1000, stop_criterion=1e-3)
print(logZ)
C = 2
sigma = 0.2

Z = np.math.factorial(C/2)*(2*sigma**2)**(C/2)
print(np.log(Z))

