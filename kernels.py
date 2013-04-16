import numpy as np
import scipy.special
from util import marshal_fn, unmarshal_fn
import copy

def gen_pairwise_matrix(f, X1, X2):
    """
    Generate the matrix M_{i,j} = f(X1(i, :), X2(j, :)).
    """

    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1,n2))
    if X1 is X2 or (X1==X2).all():
        for i in range(n1):
            for j in range(i, n2):
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

    def set_params(self, params):
        self.params = params
        self.nparams = len(params)

    def __init__(self, params, priors):
        self.priors = priors
        self.set_params(params)

    def _check_args(self, X1, X2):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        if len(X1.shape)==2 and len(X2.shape)==2 and X1.shape[1] == X2.shape[1]:
            return X1, X2
        elif len(X1.shape)==2 and X1.shape[1] == X2.shape[0]:
            return X1, np.reshape(X2, (1, -1))
        elif len(X1.shape)==1 and len(X2.shape)==1 and X1.shape[0]==X2.shape[0]:
            return np.reshape(X1, (1, X1.shape[0])), np.reshape(X2, (1, X2.shape[0]))
        elif len(X2.shape)==1 and len(X1.shape)==2 and (X1.shape[0] == 1 or X1.shape[1] == 1):
            return np.reshape(X1, (-1, 1)), np.reshape(X2, (X2.shape[0], 1))
        elif len(X1.shape)==1 and len(X2.shape)==2 and (X2.shape[0] == 1 or X2.shape[1] == 1):
            return np.reshape(X1, (X1.shape[0], 1)), np.reshape(X2, (-1, 1))
        else:
            print "HERE THEY ARE"
            print X1
            print X2
            print "THERE THEY WERE"
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

    def __call__(self, X1, X2, identical=False):
        raise RuntimeError( "Needs to be implemented in base class" )

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        raise RuntimeError( "Needs to be implemented in base class" )

    def param_prior_ll(self):
        return np.sum([b.logpdf(a) if b is not None else 0 for (a,b) in zip(self.params, self.priors)])

    def param_prior_grad(self):
        return np.asarray([b.logpdf_dx(a) if b is not None else 0 for (a,b) in zip(self.params, self.priors)])

class SumKernel(Kernel):

    def set_params(self, params):
        self.lhs.set_params(params[:self.param_split])
        self.rhs.set_params(params[self.param_split:])

    def __init__(self, lhs, rhs):
        self.lhs=lhs
        self.rhs=rhs
        self.nparams = lhs.nparams + rhs.nparams
        self.param_split = lhs.nparams

    def __call__(self, X1, X2, identical=False):
        return self.lhs(X1, X2, identical=identical)+self.rhs(X1, X2, identical=identical)

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        if i < self.lhs.nparams:
            return self.lhs.derivative_wrt_i(i, X1, X2, identical=identical)
        else:
            return self.rhs.derivative_wrt_i(i-self.lhs.nparams, X1,X2, identical=identical)

    def param_prior_ll(self):
        return self.lhs.param_prior_ll() + self.rhs.param_prior_ll()

    def param_prior_grad(self):
        g = np.concatenate([self.lhs.param_prior_grad(), self.rhs.param_prior_grad()])
        return g

class ProductKernel(Kernel):
    def set_params(self, params):
        self.lhs.set_params(params[:self.param_split])
        self.rhs.set_params(params[self.param_split:])

    def __init__(self, lhs, rhs):
        self.lhs=lhs
        self.rhs=rhs
        self.nparams = lhs.nparams + rhs.nparams
        self.param_split = lhs.nparams

    def __call__(self, X1, X2, identical=False):
        return self.lhs(X1, X2, identical=identical)*self.rhs(X1, X2, identical=identical)

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        if i < self.lhs.nparams:
            return self.lhs.derivative_wrt_i(i, X1, X2) * self.rhs(X1,X2)
        else:
            return self.rhs.derivative_wrt_i(i-self.lhs.nparams, X1,X2) * self.lhs(X1, X2)

    def param_prior_ll(self):
        return self.lhs.param_prior_ll() + self.rhs.param_prior_ll()

    def param_prior_grad(self):
        g = np.concatenate([self.lhs.param_prior_grad(), self.rhs.param_prior_grad()])
        return g


class LinearKernel(Kernel):

    def __call__(self, X1, X2, identical=False):
        return np.dot(X1, X2.T)

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        (n,d) = X1.shape
        (m,d) = X2.shape
        return np.zeros((n,m))

class SEKernel(Kernel):
    """
    Squared exponential kernel. k(x,y) = sigma2_f * exp(-||x-y||^2/ws^2)

    params[0]: sigma2_f scales the prior variance
    params[1:]: ws is an array of characteristic width scales for each coordinate.
    """

    def set_params(self, params):
        super(SEKernel, self).set_params(params)

        self.sigma2_f = params[0]
        self.ws = params[1:]
        assert( len(self.ws) > 0 )
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

    def __call__(self, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)
        wsd = self._sqdistances(X1 * self.iws, X2 * self.iws)
        K = self.sigma2_f * np.exp(-.5 * wsd)
        return K

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)
        wsd = self._sqdistances(X1*self.iws, X2*self.iws)
        if i==0:
            dK = np.exp(-.5 * wsd)
        elif i>=1 and i-1 < len(self.ws):
            dK = self.sigma2_f * np.exp(-.5 * wsd) * self.__sqidistances(i-1, X1, X2)  / (self.ws[i-1]**3)
        else:
            raise RuntimeError("Unknown parameter index %d (out of %d) for SEKernel." % (i, self.nparams))
        return dK

class DistFNKernel(Kernel):
    """
    RBF kernel using a user-supplied distance metric d.
    k(x,y) = sigma2_f * exp(-d(x,y)^2 / ws^2)
    """

    def set_params(self, params):
        super(DistFNKernel, self).set_params(params)
        if len(params) == 1:
            self.sigma2_f = 1
            self.w = params[0]
            self.df_params = []
        else:
            self.sigma2_f = params[0]
            self.w = params[1]
            self.df_params = params[2:]

    def __init__(self, params, distfn, priors = None, deriv=None):
        super(DistFNKernel, self).__init__(params, priors)

        self.distfn = lambda a,b: distfn(a,b,self.df_params)
        self.distfn_deriv_i = None if deriv is None else lambda i, a, b: deriv(i, a, b, self.df_params)

        # keep the original arguments so we can pickle/unpickle easily
        self.dfn = distfn
        self.dfn_deriv = deriv

    def __call__(self, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)

        if self.w == 0 or self.sigma2_f ==0:
            print "warning: invalid kernel parameter, returning 0", self.w, self.sigma2_f
            return np.zeros((X1.shape[0], X2.shape[0]))

        D = gen_pairwise_matrix(self.distfn, X1, X2)
        try:
            d = self.sigma2_f * np.exp(-1 * D**2 / self.w**2)
        except AttributeError:
            import pdb
            pdb.set_trace()
        return d

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)
        D = gen_pairwise_matrix(self.distfn, X1, X2)

        if self.w == 0 or self.sigma2_f ==0:
            print "warning: invalid kernel parameter, returning 0 matrix:", self.w, self.sigma2_f
            return np.zeros((X1.shape[0], X2.shape[0]))

        # import pdb; pdb.set_trace()

        if i==0 and len(self.params) > 1:
            # deriv wrt sigma2_f
            dK = np.exp(-1 * D**2 / self.w**2)
        elif i == 1 or len(self.params) == 1:
            # deriv wrt w
            dK = self.sigma2_f * np.exp(-1*D**2 / self.w**2) * 2 * D**2 / (self.w**3)
        elif i > 1:
            dD = gen_pairwise_matrix(lambda x1, x2 : self.distfn_deriv_i(i-2, x1, x2), X1, X2)
            dK = -self.sigma2_f * np.exp(-1*D**2 / self.w**2) * 2 * D / (self.w**2) * dD
        else:
            raise RuntimeError("Unknown parameter index %d (out of %d) for DistFNKernel." % (i, self.nparams))

        return dK

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        del state['distfn']
        del state['distfn_deriv_i']
        state['dfn'] = marshal_fn(self.dfn)
        if self.dfn_deriv is not None:
            state['dfn_deriv'] = marshal_fn(self.dfn_deriv)
        return state

    def __setstate__(self, state):
        state['dfn'] = unmarshal_fn(state['dfn'])
        if state['dfn_deriv'] is not None:
            state['dfn_deriv'] = unmarshal_fn(state['dfn_deriv'])
        self.__dict__.update(state)
        self.distfn = lambda a,b: self.dfn(a,b,self.df_params)
        self.distfn_deriv_i = None if self.dfn_deriv is None else \
            lambda a,b: self.dfn_deriv(a,b,self.df_params)

class SEKernelIso(SEKernel):
    """
    Squared-exponential kernel with isotropic covariance (same width
    scale for each coordinate).
    """
    def set_params(self, params):
        Kernel.set_params(self, params)
        self.sigma2_f = params[0]
        self.w = params[1]
        self.ws = None
        self.iws = None
        self.Winv = None
        self.Winv2 = None

    def init_ws(self, d):
        if self.ws is not None:
            return
        self.ws = self.w * np.ones((d,))
        self.iws = 1.0/self.w
        self.Winv = np.diag(1.0/self.ws)
        self.Winv2 = self.Winv * self.Winv

    def __call__(self, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)
        self.init_ws(X1.shape[1])
        return super(SEKernelIso, self).__call__(X1, X2)

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)
        w = self.w
        if i==0:
            dK =  np.exp(-.5 * self._sqdistances(X1/w, X2/w))
        elif i>=1 and i-1 < len(self.ws):
            wsd = self._sqdistances(X1/w, X2/w)
            dK = np.exp(-.5 * wsd) * self._sqdistances(X1/w, X2/w)/w
        else:
            raise RuntimeError("Unknown parameter index %d (out of %d) for SEKernel." % (i, self.nparams))
        return dK


class DiagonalKernel(Kernel):
    """
    Kernel given by k(x1,x2)=s^2 if x1==x2, otherwise =0. Takes a
    single parameter, s^2.
    """

    def set_params(self, params):
        super(DiagonalKernel, self).set_params(params)
        self.s2 = params[0]

    def __call__(self, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)
        (n,d) = X1.shape
        if identical:
            assert( X1.shape == X2.shape )
            return self.s2 * np.eye(n)
        else:
            (m,d) = X2.shape
            return np.zeros((n,m))
#            f = lambda x1, x2: self.s2 if x1==x2 else 0
#            return gen_pairwise_matrix(f, X1, X2)

    def derivative_wrt_i(self, i, X1, X2, identical=False):
        X1, X2 = self._check_args(X1,X2)
        if i != 0:
            raise RuntimeError("Unknown parameter index %d for DiagonalKernel." % (i))
        else:
            (n,d) = X1.shape
            if identical:
                return np.eye(n)
            else:
                (m,d) = X2.shape
                return np.zeros((n,m))

def setup_kernel(name, params, extra=None, priors=None):
    """
    Construct a kernel object from a string description.
    """

    if priors is None:
        priors = [None for p in params]

    if name == "linear":
        sigma2_n = params[0]
        k = DiagonalKernel([sigma2_n,], priors = priors[0]) + LinearKernel(params[1:], params[1:])
    elif name == "se":
        sigma2_n = params[0]
        k = DiagonalKernel([sigma2_n,], priors = priors[0:1]) + SEKernel(params[1:], priors[1:])
    elif name == "se_iso":
        sigma2_n = params[0]
        k = DiagonalKernel([sigma2_n,], priors = priors[0:1]) + SEKernelIso(params[1:], priors[1:])
    elif name == "se_noiseless":
        k = SEKernel(params)
    elif name == "distfn":
        sigma2_n = params[0]
        sigma2_f = params[1]
        w = params[2]
        if len(params) > 3:
            distfn = extra[0]
            distfn_deriv_i = extra[1]
        else:
            distfn = extra
            distfn_deriv_i = None
        distfn_params = list(params[3:])

        k = DiagonalKernel([sigma2_n,], priors = np.asarray(priors[0:1])) + DistFNKernel([sigma2_f, w] + distfn_params, distfn, priors = priors[1:], deriv = distfn_deriv_i)

    elif name == "distfns_sum":
        # composite kernel of the form sum_i alpha_i *
        # exp(-r_i(x1,x2)^2 / beta_i^2) where each r_i is a distance
        # function and alpha_i and beta_i are magnitude and
        # length-scale parameters respectively.

        # here "params" is a list of 2n+1 entries. The first entry is
        # sigma2_n. The ith subsequent pair of entries gives alpha and
        # beta for the ith distance function. "priors" has the same
        # structure and semantics as "params".
        pass
        # "extra" is a list of n tuples of the form (distfn,
        # distfn_deriv). distfn_deriv can be None if there are no
        # special params to the distance function.

        sigma2_n = params[0]
        composite_kernel = DiagonalKernel([sigma2_n,], priors = [priors[0],])
        for (i, (distfn, distfn_deriv_i)) in enumerate(extra):
            si = 2*i+1
            kparams = params[si:si+2]
            kpriors = priors[si:si+2]
            k = DistFNKernel(kparams, distfn, priors = kpriors, deriv = distfn_deriv_i)
            composite_kernel += k
        k = composite_kernel

    elif name == "distfns_prod":
        # similar to above, except that we construct a product of
        # distfn kernels, so there is only a single sigma2_f magnitude
        # parameter (specified after sigma2_n) instead of an alpha param
        # for each kernel.

        sigma2_n = params[0]
        noise_kernel = DiagonalKernel([sigma2_n,], priors = [priors[0], ])

        sigma2_f = params[1]
        composite_kernel = DistFNKernel([sigma2_f, 1], lambda a,b,p : 0, priors = [priors[1],], deriv = None)
        for (i, (distfn, distfn_deriv_i)) in enumerate(extra):
            si = i+2
            kparams = params[si:si+1]
            kpriors = priors[si:si+1]
            k = DistFNKernel(kparams, distfn, priors = kpriors, deriv = distfn_deriv_i)
            composite_kernel *= k
        k = composite_kernel + noise_kernel
    elif name == "composite":

        # assume we are passed the following functions in kernel_extra:
        # ke[0] = dist_diff  -- returns difference of log(ev-to-sta) distances in KM
        # ke[1] = azi_diff   -- returns differences in azimuth, in degrees (always positive)
        # ke[2] = depth_diff -- returns difference of log(depth) in KM
        # ke[3] = local_dist -- returns distance in KM between two events (including depth)

        # assume we are passed the following params/priors:
        # 0 : sigma2_n -- noise variance
        # 1 : sigma2_f_dist -- function variance wrt dist_diff
        # 2 : w_dist -- length scale for dist_diff
        # 3 : sigma2_f_azi -- function variance wrt azi_diff
        # 4 : w_azi -- length scale for azi_diff
        # 5 : sigma2_f_depth -- function variance wrt depth_diff
        # 6 : w_depth -- length scale for depth_diff
        # 7 : sigma2_f_local -- function variance wrt local_dist
        # 8 : w_local -- length scale for local_dist
        noise_kernel = DiagonalKernel(params=params[0:1], priors = priors[0:1])
        distdiff_kernel = DistFNKernel(params=params[1:3], priors=priors[1:3],
                                       distfn = extra[0], deriv=None)
        azidiff_kernel = DistFNKernel(params=params[3:5], priors=priors[3:5],
                                      distfn = extra[1], deriv=None)
        depthdiff_kernel = DistFNKernel(params=params[5:7], priors=priors[5:7],
                                        distfn = extra[2], deriv=None)
        local_kernel = DistFNKernel(params=params[7:9], priors=priors[7:9],
                                    distfn = extra[3], deriv=None)
        k = noise_kernel + distdiff_kernel + azidiff_kernel + depthdiff_kernel + local_kernel
    else:
        raise RuntimeError("unrecognized kernel name %s." % (name))
    return k
