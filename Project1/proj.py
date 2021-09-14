import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee
import corner

sns.set()
sns.set_context("talk")


# prepare data
x, data, sigma = np.loadtxt("D1_c_5.dat", unpack=True)
m = len(x)
x = x.reshape(m,1)
data = data.reshape(m,1)
sigma = sigma.reshape(m,1)

k_max = 2

min_theta = np.ones(k_max) * -100
max_theta = np.ones(k_max) * 100
volume_theta = np.prod(max_theta-min_theta)


def g(x, theta):
    """ """
    return np.sum([theta[i]*x**i for i in range(len(theta))], axis=0)

def chi_squared(x, data, sigma, g, theta):
    """ """
    return np.sum( (data - g(x, theta))**2/sigma**2 )


def log_prior_uniform(theta):
    """Stuff
    """
    # flat prior 
    if np.logical_and(min_theta<=theta, theta<=max_theta).all(): 
        return np.log(1/volume_theta)
    else:
        return -np.inf

def log_prior_gaussian(theta, abar):
    """Stuff and neglect normalistation.
    """
    k = len(theta)
    return - np.sum(theta**2)/(2*abar**2)
    

def log_likelihood(theta, X):
    '''Log likelihood for data X given parameter array theta'''
    try:
        return -0.5 * np.sum( ( (X - theta[0]) / theta[1] )** 2 ) \
               - 0.5*len(X)*np.log(2*np.pi*theta[1]**2)
    except ValueError:
        return -np.inf

def log_likelihood(x, data, sigma, g, theta):
    '''Log likelihood for data X given parameter array theta'''
    try:
        return -chi_squared(x, data, sigma, g, theta)/2
    except ValueError:
        return -np.inf


def log_posterior(theta, X, log_prior=log_prior_uniform):
    '''Log posterior for data X given parameter array theta'''
    return log_prior(theta) + log_likelihood(x, data, sigma, g, theta)


def bayesian_inference(k, nwalkers=50, nburn=1000, nsteps=1000):
    ndim = k  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 1000  # number of MCMC steps to take

    # we'll start at random locations within the prior volume
    starting_guesses = min_theta + max_theta * np.random.rand(nwalkers,ndim)

    print("MCMC sampling using emcee (affine-invariant ensamble sampler) with {0} walkers".format(nwalkers))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])
    
    # "burn-in" period; save final positions and then reset
    pos, prob, state = sampler.run_mcmc(starting_guesses, nburn)
    sampler.reset()

    # sampling period
    sampler.run_mcmc(pos, nsteps)

    print("Mean acceptance fraction: {0:.3f} (in total {1} steps)"
          .format(np.mean(sampler.acceptance_fraction),nwalkers*nsteps))

    # discard burn-in points and flatten the walkers;
    # the shape of samples is (nwalkers*nsteps, ndim)
    samples = sampler.chain.reshape((-1, ndim))


    mu_true = 1
    sigma_true = 2
    
    # make a corner plot with the posterior distribution
    fig = corner.corner(samples, labels=["$\mu$", "$\sigma$"],
                        truths=[mu_true, sigma_true],quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12})

    # With some manual efforts, we can add the maximum-likelihood estimate from the frequentist analysis
    mu_est = 1
    sigma_est =2
    maxlike_results = (mu_est,sigma_est)

    # First, extract the axes
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Then, loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(maxlike_results[i], color="r")

    # And finally, loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(maxlike_results[xi], color="r")
            ax.axhline(maxlike_results[yi], color="r")
            ax.plot(maxlike_results[xi], maxlike_results[yi], "sr")

    plt.show()



bayesian_inference(2)
