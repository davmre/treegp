import numpy as np
import scipy.special

def gen_pairwise_matrix(f, X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1,n2))
    if X1 is X2:
        for i in range(n1):
            for j in range(i, n2):
#                print X1[i, :], X2[j,:]
                K[i, j] = f(X1[i,:], X2[j,:])
        K = K + K.T - np.diag(np.diag(K))
    else:
        for i in range(n1):
            for j in range(n2):
                K[i, j] = f(X1[i,:], X2[j,:])
    return K

def almost_equal(x1, x2, tol=1e-6):
    return np.linalg.norm(x1-x2) < tol

# abstract base class for kernels
class Kernel(object):

    def __init__(self, params, priors):
        self.params = params
        self.nparams = len(params)
        self.priors = priors

    def _check_args(self, X1, X2):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        if len(X1.shape)==2 and len(X2.shape)==2 and X1.shape[1] == X2.shape[1]:
            return X1, X2
        elif len(X1.shape)==2 and X1.shape[1] == X2.shape[0]:
            return X1, np.reshape(X2, (1, -1))
        elif len(X1.shape)==1 and len(X2.shape)==1 and X1.shape[0]==X2.shape[0]:
            return np.reshape(X1, (1, X1.shape[0])), np.reshape(X2, (1, X2.shape[0]))
        else:
            raise RuntimeError("Incompatible dimensions in kernel arguments.")

    def __mul__(self, k_rhs):
        """
        @return: The product of self and k_rhs
        """
        return ProductKernel(self, k_rhs)

    def __add__(self, k_rhs):
        """
        @return: The sum of self and k_rhs
        """
        return SumKernel(self, k_rhs)

    def __call__(self, X1, X2):
        raise RuntimeError( "Needs to be implemented in base class" )

    def derivative_wrt_i(self, i, X1, X2):
        raise RuntimeError( "Needs to be implemented in base class" )

    def param_prior_ll(self):
        return np.sum([b.logpdf(a) for (a,b) in zip(self.params, self.priors)])

    def param_prior_grad(self):
        return np.asarray([b.logpdf_dx(a) for (a,b) in zip(self.params, self.priors)])

class SumKernel(Kernel):
    def __init__(self, lhs, rhs):
        self.lhs=lhs
        self.rhs=rhs
        self.nparams = lhs.nparams + rhs.nparams
        self.param_split = lhs.nparams

    def __call__(self, X1, X2):
        return self.lhs(X1, X2)+self.rhs(X1, X2)

    def derivative_wrt_i(self, i, X1, X2):
        if i < self.lhs.nparams:
            return self.lhs.derivative_wrt_i(i, X1, X2)
        else:
            return self.rhs.derivative_wrt_i(i-self.lhs.nparams, X1,X2)

    def param_prior_ll(self):
        return self.lhs.param_prior_ll() + self.rhs.param_prior_ll()

    def param_prior_grad(self):
        g = np.concatenate([self.lhs.param_prior_grad(), self.rhs.param_prior_grad()])
        return g


class SEKernel(Kernel):

    def __init__(self, params, priors = None):
        super(SEKernel, self).__init__(params, priors)
        self.sigma_f = params[0]
        self.ws = params[1:]
        self.iws = 1/self.ws
        self.Winv = 1/np.diag(self.ws)
        self.Winv2 = self.Winv*self.Winv

    def _sqdistances(self, X1, X2):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        norms1 = np.reshape(np.sum(np.abs(X1)**2,axis=-1), (n1, 1))
        norms2 = np.reshape(np.sum(np.abs(X2)**2,axis=-1), (n2, 1))
        sqdistances = np.tile(norms1, (1, n2)) + np.tile(norms2.T, (n1, 1)) - 2*np.dot(X1, X2.T)

        return sqdistances

    def __sqidistances(self, i, X1, X2):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        iX1 = np.tile(np.reshape(X1[:, i], (-1, 1)), (1, n2))
        iX2 = np.tile(np.reshape(X2[:, i], (1, -1)), (n1, 1))
#        print iX1.shape, iX2.shape

        sqi = (iX1 - iX2)**2
        return sqi

    def __call__(self, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        wsd = self._sqdistances(X1 * self.iws, X2 * self.iws)
        K = self.sigma_f * np.exp(-.5 * wsd)
        return K

    def derivative_wrt_i(self, i, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        wsd = self._sqdistances(X1*self.iws, X2*self.iws)
        if i==0:
            dK = np.exp(-.5 * wsd)
        elif i>=1 and i-1 < len(self.ws):
            dK = self.sigma_f * np.exp(-.5 * wsd) * self.__sqidistances(i-1, X1, X2)  / (self.ws[i-1]**3)
        else:
            raise RuntimeError("Unknown parameter index %d (out of %d) for SEKernel." % (i, self.nparams))
        return dK


class DistFNKernel(Kernel):
    def __init__(self, params, distfn, priors = None):
        super(DistFNKernel, self).__init__(params, priors)
        self.sigma_f = params[0]
        self.w = params[1]
        self.distfn = distfn

    def __call__(self, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        D = gen_pairwise_matrix(self.distfn, X1, X2)
        return self.sigma_f**2 * np.exp(-1 * D**2 / self.w**2)

    def derivative_wrt_i(self, i, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        D2 = gen_pairwise_matrix(self.distfn, X1, X2)**2
        if i==0:
            dK = 2*self.sigma_f * np.exp(-1 * D2 / self.w**2)
        elif i == 1:
            dK = 2 * self.sigma_f**2 * np.exp(-1*D2 / self.w**2) * D2 / (self.w**3)
        else:
            raise RuntimeError("Unknown parameter index %d (out of %d) for DistFNKernel." % (i, self.nparams))

        return dK
        

class SEKernelIso(SEKernel):
    def __init__(self, params, priors = None):
        super(SEKernel, self).__init__(params, priors)
        self.sigma_f = params[0]
        self.sigma2_f = params[0]**2
        self.w = params[1]
        self.ws = None
        self.iws = None
        self.Winv = None
        self.Winv2 = None

    def init_ws(self, d):
        if self.ws is not None:
            return
        self.ws = self.w * np.ones((d,))
        self.iws = 1/self.w
        self.Winv = 1/np.diag(self.ws)
        self.Winv2 = self.Winv * self.Winv

    def __call__(self, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        self.init_ws(X1.shape[1])
        return super(SEKernelIso, self).__call__(X1, X2)

    def derivative_wrt_i(self, i, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        w = self.w
        if i==0:
            dK = 2 * self.sigma_f * np.exp(-.5 * self._sqdistances(X1/w, X2/w))
        elif i>=1 and i-1 < len(self.ws):
            wsd = self._sqdistances(X1/w, X2/w)
            dK = self.sigma2_f * np.exp(-.5 * wsd) * self._sqdistances(X1/w, X2/w)/w
        else:
            raise RuntimeError("Unknown parameter index %d (out of %d) for SEKernel." % (i, self.nparams))
        return dK


class DiagonalKernel(Kernel):
    """
    Kernel given by k(x1,x2)=s^2 if x1==x2, otherwise =0. Takes a
    single parameter, s.
    """
    def __init__(self, params, priors=None):
        super(DiagonalKernel, self).__init__(params, priors)
        self.s2 = self.params[0]**2
        self.s = self.params[0]

    def __call__(self, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        if X1 is X2:
            (n,d) = X1.shape
            return self.s2 * np.eye(n)
        else:
            f = lambda x1, x2: self.s2 if almost_equal(x1,x2) else 0
            return gen_pairwise_matrix(f, X1, X2)

    def derivative_wrt_i(self, i, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        if i != 0:
            raise RuntimeError("Unknown parameter index %d for DiagonalKernel." % (i))
        else:
            if X1 is X2:
                (n,d) = X1.shape
                return 2 * self.s * np.eye(n)
            else:
                f = lambda x1, x2: 2*self.s if almost_equal(x1,x2) else 0
                return gen_pairwise_matrix(f, X1, X2)

def setup_kernel(name, params, extra, priors=None):
    if priors is None:
        priors = [None for p in params]

    if name == "se":
        sigma_n = params[0]
        k = DiagonalKernel([sigma_n,], priors = priors[0:1]) + SEKernel(params[1:], priors[1:])
    elif name == "se_iso":
        sigma_n = params[0]
        k = DiagonalKernel([sigma_n,], priors = priors[0:1]) + SEKernelIso(params[1:], priors[1:])
    elif name == "se_noiseless":
        k = SEKernel(params)
    elif name == "distfn":
        sigma_n = params[0]
        sigma_f = params[1]
        w = params[2]
        distfn = extra
        k = DiagonalKernel([sigma_n,], priors = np.asarray(priors[0:1])) + DistFNKernel([sigma_f, w], distfn, priors = priors[1:])
    else:
        raise RuntimeError("unrecognized kernel name %s." % (name))
    return k


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
        if 0.0 >= x: return np.log(1e-300)
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

