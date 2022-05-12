import numpy as np
import matplotlib.pyplot as plt

def likelihood(x, C, sigma):
    return np.exp(-x**(2/C)/(2*sigma**2))

def prior(C, n_samples):
    #random_directions = np.random.normal(size=(C,n_samples))
    #norm = np.linalg.norm(random_directions, axis=0)
    #random_directions/=norm
    radii = np.random.random(n_samples)**(1/C)
    return radii**2

def nested_sampling(likelihood, prior, n_dim, nlive, stop_criterion, sigma):
    #initialisation
    Z = 0
    X_previous = 1
    iteration = 0
    increase = 1000

    #sample from prior
    samples = prior(n_dim, nlive)
    likelihoods = likelihood(samples, n_dim, sigma=sigma)
    samples = samples.tolist()
    likelihoods = likelihoods.tolist()


    while increase > stop_criterion:
        iteration += 1
        minLike = min(likelihoods)
        index = likelihoods.index(minLike)

        X_current = np.exp(-iteration/nlive)
        weight_current = X_previous- X_current
        X_previous = X_current

        Z += weight_current*minLike


        sampling = True
        while sampling:
            proposal_sample = prior(n_dim,1)

            if likelihood(proposal_sample,n_dim, sigma=sigma) > minLike:
                #accept
                samples[index] = float(proposal_sample)
                likelihoods[index] = float(likelihood(proposal_sample,n_dim, sigma=sigma))
                sampling = False

        maxLike = max(likelihoods)
        #print(np.log10(maxLike))
        #print(np.log10(Z))
        increase = weight_current*maxLike/Z
        if iteration%1000 == 0:
            print("current iteration: ", iteration)
            #print("current increase: ", increase)

    finalLikelihoods = likelihood(np.array(samples), n_dim, sigma=sigma)
    Z += nlive**(-1)*np.sum(finalLikelihoods)*X_current

    return Z


Z =nested_sampling(likelihood=likelihood, prior=prior, n_dim=2,nlive=1000, stop_criterion=1e-3, sigma=0.01)
print(np.log(Z))
C = 2
sigma = 0.01

Z = np.math.factorial(C/2)*(2*sigma**2)**(C/2)
print(np.log(Z))

