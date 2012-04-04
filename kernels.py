import numpy as np


def gen_pairwise_matrix(f, X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1,n2))
    if X1 is X2:
        for i in range(n1):
            for j in range(i, n2):
                K[i, j] = self.f(X1[i,:], X2[j:])
        K = K + K.T - np.diag(np.diag(K))
    else:
        for i in range(n1):
            for j in range(n2):
                K[i, j] = f(X1[i,:], X2[j:])
    return K

def almost_equal(x1, x2, tol=1e-6):
    return np.linalg.norm(x1-x2) < tol

# abstract base class for kernels
class Kernel(object):

    def __init__(self, params):
        self.params = params
        self.nparams = len(params)

    def _check_args(self, X1, X2):
        if len(X1.shape)==2 and len(X2.shape)==2 and X1.shape[1] == X2.shape[1]:
            return X1, X2
        elif len(X1.shape)==2 and X1.shape[1] == X2.shape[0]:
            return X1, np.reshape(X2, (1, -1))
        elif len(X1.shape)==1 and len(X2.shape)==1 and X1.shape[0]==X2.shape[0]:
            return np.reshape(X1, (1, X1.shape[0])), np.reshape(X1, (1, X1.shape[0]))
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

class ProductKernel(Kernel):
    def __init__(self, lhs, rhs):
        self.lhs=lhs
        self.rhs=rhs
        self.nparams = lhs.nparams+rhs.nparams

    def __call__(self, X1, X2):
        return self.lhs(X1, X2)*self.rhs(X1, X2)

    def derivative_wrt_i(self, i, X1, X2):
        if i < self.lhs.nparams:
            return self.lhs.derivative_wrt_i(i, X1, X2)*self.rhs(X1,X2)
        else:
            return self.lhs(X1, X2)*self.rhs.derivative_wrt_i(i-self.lhs.nparams, X1,X2)

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

class SEKernel(Kernel):

    def __init__(self, params):
        super(SEKernel, self).__init__(params)
        self.sigma_f = params[0]
        self.sigma2_f = params[0]**2
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
        print iX1.shape, iX2.shape

        sqi = (iX1 - iX2)**2
        return sqi

    def __call__(self, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        K = self.sigma2_f * np.exp(-.5 * self._sqdistances(X1 * self.iws, X2 * self.iws))
        return K

    def derivative_wrt_i(self, i, X1, X2):
        X1, X2 = self._check_args(X1,X2)
        if i==0:
            dK = 2 * self.sigma_f * np.exp(-.5 * self._sqdistances(X1*self.iws, X2*self.iws))
        elif i>=1 and i-1 < len(self.ws):
            wsd = self._sqdistances(X1*self.iws, X2*self.iws)
            dK = self.sigma2_f * np.exp(-.5 * wsd) * self.__sqidistances(i-1, X1, X2)  / (self.ws[i-1]**3)
        else:
            raise RuntimeError("Unknown parameter index %d (out of %d) for SEKernel." % (i, self.nparams))
        return dK

class SEKernelIso(SEKernel):
    def __init__(self, params):
        super(SEKernel, self).__init__(params)
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
    def __init__(self, params):
        super(DiagonalKernel, self).__init__(params)
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

def setup_kernel(kernel_name, kernel_params):
    if kernel_name == "se":
        sigma_n = kernel_params[0]
        k = DiagonalKernel([sigma_n,]) + SEKernel(kernel_params[1:])
    elif kernel_name == "se_iso":
        sigma_n = kernel_params[0]
        k = DiagonalKernel([sigma_n,]) + SEKernelIso(kernel_params[1:])
    elif kernel_name == "se_noiseless":
        k = SEKernel(kernel_params)
    else:
        raise RuntimeError("unrecognized kernel name %s." % (kernel_name))
    return k

