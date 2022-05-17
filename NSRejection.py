import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.special
def logLikelihood(x, ndim):
    #Multivariate Gaussian centred at X = 0.5, y= 0.5
    #x shape: (ndim, n_samples)
    means = 0.5*np.ones(shape=ndim)
    cov = 0.05*np.eye(N=ndim)
    return multivariate_normal.logpdf(x=x, mean=means, cov=cov)



def prior(ndim, nsamples):
    #random_directions = np.random.normal(silogze=(C,n_samples))
    #norm = np.linalg.norm(random_directions, axis=0)
    #random_directions/=norm
    return np.random.uniform(low=0, high=1, size=(nsamples,ndim))

def nested_sampling(logLikelihood, prior, ndim, nlive, stop_criterion):
    #initialisation
    logZ_previous = -1e300 # Z = 0
    logX_previous = 0 # X = 1
    iteration = 0
    logIncrease = 10 # evidence increase factor

    #sample from prior
    samples = prior(ndim, nlive)
    logLikelihoods = logLikelihood(samples, ndim)
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

        logZ_current = logWeight_current + minlogLike
        logZ_array = np.array([logZ_previous, logZ_current])
        logZ_total =  scipy.special.logsumexp(logZ_array)
        logZ_previous = logZ_total


        sampling = True
        while sampling:
            proposal_sample = prior(ndim,1)
            if logLikelihood(proposal_sample,ndim) > minlogLike:
                #accept
                if ndim > 1:
                    samples[index] = proposal_sample.tolist()[0]
                else:
                    samples[index] = float(proposal_sample)
                logLikelihoods[index] = float(logLikelihood(proposal_sample,ndim))
                sampling = False

        maxlogLike = max(logLikelihoods)
        logIncrease = logWeight_current+maxlogLike-logZ_total
        if iteration%1000 == 0:
            print("current iteration: ", iteration)
            #print("current increase: ", increase)
    samples = np.array(samples)
    finallogLikesum = scipy.special.logsumexp(a=logLikelihoods)
    logZ_current = -np.log(nlive) + finallogLikesum + logX_current
    logZ_array = np.array([logZ_previous, logZ_current])
    logZ_total = scipy.special.logsumexp(logZ_array)


    return logZ_total


logZ =nested_sampling(logLikelihood=logLikelihood, prior=prior, ndim=2,nlive=1000, stop_criterion=1e-3)
print(logZ)
C = 2
sigma = 0.2

#Z = np.math.factorial(C/2)*(2*sigma**2)**(C/2)
#print(np.log(Z))


