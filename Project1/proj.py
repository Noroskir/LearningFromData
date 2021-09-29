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
truths = [1/4, np.pi/2, np.pi**2/4,np.pi**3/24, np.pi**4/24, np.pi**5/240, 17*np.pi**6/2880]
abar = 5

def g(x, theta):
    """ """
    return np.sum([theta[i]*x**i for i in range(len(theta))], axis=0)

def chi_squared(x, data, sigma, theta):
    """Calculate chi^2 from the data given the parameters theta.
    Args:
        x     (np.array): x-values of the measurements, shape (m, 1)
        data  (np.array): y-values of the measurements, shape (m, 1)
        sigma (np.array): error values of the measurements, shape (m, 1)
    Returns:
        float: the chi^2 values. """
    g_vals = np.sum([theta[i]*x**i for i in range(len(theta))], axis=0)
    return np.sum( (data - g_vals)**2/sigma**2 )


def log_prior_uniform(theta, max_t=100):
    """Flat uniform log prior for the range [-max_t, max_t]
    Args:
        theta (np.array): array of the coefficients.
        max_t (float, optional): maximal absolute value for theta. 
    Returns:
        np.array: the log of the uniform prior."""
    min_theta = np.ones(len(theta)) * -max_t
    max_theta = np.ones(len(theta)) * max_t
    volume_theta = np.prod(max_theta-min_theta)
    if np.logical_and(min_theta<=theta, theta<=max_theta).all(): 
        return 0
    else:
        return -np.inf

def log_prior_gaussian(theta, abar=5):
    """Stuff and neglect normalistation.
    """
    k = len(theta)
    return - np.sum(theta**2)/(2*abar**2)
    


def log_likelihood(x, data, sigma, g, theta):
    '''Log likelihood for data X given parameter array theta'''
    try:
        return np.sum(-np.log(np.sqrt(2*np.pi*sigma))) - chi_squared(x, data, sigma, theta)/2
    except ValueError:
        print("value error")
        return -np.inf


def log_posterior(theta, x, data, sigma, log_prior=log_prior_uniform, max_t=100):
    '''Log posterior for data X given parameter array theta'''
    if log_prior == log_prior_uniform:
        return log_prior(theta, max_t) + log_likelihood(x, data, sigma, g, theta)
    else:
        return log_prior(theta) + log_likelihood(x, data, sigma, g, theta)


def sample_probabilities(k, x, data, sigma, prior=log_prior_uniform, nwalkers=50, 
                         nburn=1000, nsteps=1000, max_t=100):
    k_max = k

    min_theta = np.ones(k_max) * -max_t
    max_theta = np.ones(k_max) * max_t
    volume_theta = np.prod(max_theta-min_theta)

    ndim = k  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers

    # we'll start at random locations within the prior volume
    starting_guesses = min_theta + max_theta * np.random.rand(nwalkers,ndim)

    print("MCMC sampling using emcee (affine-invariant ensamble sampler) with {0} walkers".format(nwalkers))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, data, sigma, prior, max_t])
    
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
    print("Shape of samples: ",samples.shape)
    

    return samples

def plot_parameters(k, samples, truths, param=0):
    # make a corner plot with the posterior distribution
    if not param or param > k:
        k_plot = k
    else:
        k_plot = param
    
    # TODO: remove
    k_plot = k
    
    labels = [r'$a_{:}$'.format(i) for i in range(len(truths))]
    fig = corner.corner(samples[:,:k_plot], labels=labels[:k_plot],
                        truths=truths[:k_plot],quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12})


    # First, extract the axes
    axes = np.array(fig.axes).reshape((k_plot, k_plot))
    plt.show()


# ==================================================
# Gaussian prior
# ==================================================

def analysis_gaussian(x, data, sigma, k_max):
    samples_normal = []
    params_normal = []

    for k in range(1,k_max+2):
        print(f"k_max = {k-1}")
        samples = sample_probabilities(k, x, data, sigma, prior=log_prior_gaussian,
                                       nburn=1000, nsteps=5000)
        samples_normal.append(samples)
        m, a, p = np.percentile(samples, [16, 50, 84], axis=0)
        params_normal.append([a,m,p])
        for i in range(k):
            print("a_{:} = {:2.4f} + {:2.4f} - {:2.4f}".format(i, m[i],a[i],p[i]))
        print("")
    return samples_normal, params_normal


def calc_evidence(x, data, sigma, theta, s_theta):
    k = len(theta)
    sum_sigma = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            sum_sigma[i,j] = np.sum(1/sigma**2 * x**i * x**j)
            if i == j: sum_sigma[i,j] += 1/abar**2
    det_sigma = np.linalg.det(sum_sigma)
    #print("det_sigma: ", det_sigma)
    #det_sigma = np.prod(s_theta)
    #lamb = np.prod(s_theta)
    Z = np.exp(-1/2* (chi_squared(x, data, sigma, theta) + np.sum(theta**2)/abar**2))
    Z = Z * np.sqrt((2*np.pi)**k/det_sigma)
    Z = Z * np.prod(1/np.sqrt(2*np.pi*sigma**2)) * np.prod(1/np.sqrt(2*np.pi*abar**2))
    return Z 

def eval_evidence(x, data, sigma, params):
    k = len(params)
    for i in range(k):
        a, m, p = params[i]
        s_a = (p-m)/2
        evid = calc_evidence(x, data, sigma, a, s_a)
        print(f"k_max = {i}, evidence = {evid:.3g}")
        
        
