import numpy as np
import scipy

class Distribution(object):
    def __init__(self):
        pass
    def logpdf(x):
        raise RuntimeError("method not implemented by child class")
    def logpdf_grad(x):
        raise RuntimeError("method not implemented by child class")


class Gamma(Distribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def logpdf(self, x):
        alpha = self.alpha
        beta = self.beta

        if x < 0.0: return np.log(1e-300)
        # the special case of an exponential distribution is defined even when x==0
        if alpha == 1: return np.log(beta) - beta*x
        if x == 0.0: return np.log(1e-300)
        lp = alpha*np.log(beta) - scipy.special.gammaln(alpha) + (alpha-1)*np.log(x) - beta*x
        if np.isnan(lp):
            lp = np.float("-inf")
        return lp

    def logpdf_dx(self, x):
        alpha = self.alpha
        beta = self.beta
        return (alpha-1)/x - beta


class InvGamma(Distribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def logpdf(self, x):
        alpha = self.alpha
        beta = self.beta
        lp = alpha*np.log(beta) - scipy.special.gammaln(alpha) - (alpha+1)*np.log(x) - beta/x
        if np.isnan(lp):
            lp = np.float("-inf")

        return lp

    def logpdf_dx(self, x):
        alpha = self.alpha
        beta = self.beta
        return beta/(x**2) - (alpha+1)/x

class LogNormal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def logpdf(self, x):
        mu = self.mu
        sigma = self.sigma
        lp = -1 * np.log(x) - .5 * np.log(2*np.pi*sigma) - .5 * (np.log(x) - mu)**2 / sigma**2
        if np.isnan(lp):
            lp = np.float("-inf")
        return lp

    def logpdf_dx(self, x):
        mu = self.mu
        sigma = self.sigma
        return (-1 -(np.log(x) - mu)/(sigma**2)) / x

