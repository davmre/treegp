from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sys
import itertools, pickle, traceback

import numpy as np
import scipy.linalg, scipy.optimize

from gp import GaussianProcess
import kernels

def gp_ll(X, y, kernel, kernel_params, kernel_extra):
    """
    Get the training data log-likelihood for a choice of kernel params.
    """
    try:
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra)
        ll = gp.log_likelihood()
    except:
        ll = np.float("-inf")
    return ll

def gp_grad(X, y, kernel, kernel_params, kernel_extra):
    """
    Get the training data log-likelihood gradient for a choice of kernel params.
    """

    try:
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra)
        grad = gp.log_likelihood_gradient()
    except:
        grad = np.zeros(kernel_params.shape)
    return grad

def gp_nll_ngrad(X, y, kernel, kernel_params, kernel_extra, kernel_priors):
    """
    Get both the negative log-likelihood and its gradient
    simultaneously (more efficient than doing it separately since we
    only create one new GP object, which only constructs the kernel
    matrix once, etc.).
    """
    try:
        print "optimizing params", kernel_params
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra, kernel_priors=kernel_priors)

        nll = -1 * gp.log_likelihood()
        ngrad = -1 * gp.log_likelihood_gradient()

        npll = -1 * gp.kernel.param_prior_ll()
        npgrad = -1 * gp.kernel.param_prior_grad()

#        print "nll %f + %f = %f" % (nll, npll, nll+npll)

        nll += npll
        ngrad += npgrad

        print "   ...grad", -1*ngrad
        print "   ...ll", -1 *nll,

    except np.linalg.linalg.LinAlgError as e:
        print "warning: lin alg error (%s) in likelihood computation, returning likelihood -inf" % str(e)
        nll = np.float("inf")
        ngrad = np.zeros(kernel_params.shape)
    except ValueError as e:
        print "warning: value error (%s) in likelihood computation, returning likelihood -inf" % str(e)
        nll = np.float("inf")
        ngrad = np.zeros(kernel_params.shape)

    return nll, ngrad

def learn_hyperparams(X, y, kernel, start_kernel_params, kernel_extra=None, kernel_priors=None, random_starts=0):
    """
    Use L-BFGS to search for the maximum likelihood kernel hyperparams.
    """

    ll = lambda params: -1 * gp_ll(X, y, kernel, params, kernel_extra)
    grad = lambda params: -1 * gp_grad(X, y, kernel, params, kernel_extra)
#    best_params = scipy.optimize.fmin_bfgs(f=ll, x0=start_kernel_params, fprime=grad)

    llgrad = lambda params: gp_nll_ngrad(X, y, kernel, params, kernel_extra, kernel_priors)

    skp = np.asarray(start_kernel_params)
    new_params = lambda  :  np.exp(np.log(skp) + np.random.randn(len(skp)) * 2)
    start_param_set = [skp,] + [new_params() for i in range(random_starts)]

    print "start param set"
    print start_param_set

    best_params = skp
    nll, grad = llgrad(skp)
    best_cost = nll

    for params in start_param_set:

#    best_params, v, d = scipy.optimize.fmin_l_bfgs_b(func=llgrad, x0=start_kernel_params, bounds= [(1e-20, None),]*len(start_kernel_params))
        opt_params, v, d = scipy.optimize.fmin_l_bfgs_b(func=llgrad, x0=start_kernel_params, bounds= [(1e-20, None),]*len(start_kernel_params))

        if v < best_cost:
            best_cost = v
            best_params = opt_params

#    print "start ll", ll(start_kernel_params)
#    print "best ll", ll(best_params)

    print "OPTIMZIATION FINISHED: found best params", best_params
    print "ll", v
    return np.asarray(best_params), v
