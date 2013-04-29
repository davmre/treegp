import sys
import hashlib
import itertools, pickle, traceback

import numpy as np
import scipy.linalg, scipy.optimize

import kernels
from kdtree import KDTree

from util import marshal_fn, unmarshal_fn

def log_det(X):
    w = np.linalg.eigvalsh(X)
    return np.sum(np.log(w))

class GaussianProcess:

    def __init__(self, X=None, y=None,
                 kernel = None, K=None,
                 mean="constant", fname=None,
                 basisfns=None, param_mean=None, param_cov=None,
                 compute_grad=False,
                 save_extra_info=False):

        """ Initialize a Gaussian process by providing EITHER the training data (X, y) and kernel info (kernel, kernel_params, kernel_priors, kernel_extra) OR  the filename of a serialized GP.
        'K' is an optional precomputed kernel matrix (for efficiency).
        'mean' specifies a method for estimating the function mean ("zero", "constant", or "linear").
        """

        if fname is not None:
            self.load_trained_model(fname)
        else:
            self.kernel = kernel
            self.X = X
            self.n = X.shape[0]

            #self.tree=None
            #if kdtree:
            #    self.tree = KDTree(pts=X)

            self.basisfns = None
            if mean == "zero":
                self.mu = 0
                self.y = y
            if mean == "constant":
                self.mu = np.mean(y)
                self.y = y - self.mu
            if mean == "parametric":
                self.mu = 0
                self.basisfns = basisfns
                H = np.array([[f(x) for x in X] for f in basisfns], dtype=float)
                b = param_mean
                B = param_cov
                self.y = y
                if save_extra_info:
                    self.H = H

            # train model
            if K is None:
                K = self.kernel(X, X, identical=True)
                K += np.eye(K.shape[0]) * 1e-8 # try to avoid losing
                                               # positive-definiteness
                                               # to numeric issues

            L = None
            self.alpha = None
            self.Kinv = None
            try:
                L = scipy.linalg.cholesky(K, lower=True)
                self.alpha = scipy.linalg.cho_solve((L, True), self.y)
                self.invL = scipy.linalg.inv(L)
                if save_extra_info:
                    self.L = L
            except np.linalg.linalg.LinAlgError:
                #u,v = np.linalg.eig(K)
                #print K, u
                #import pdb; pdb.set_trace()
                raise
            except ValueError:
                raise

            if self.basisfns:
                # notation follows section 2.7 of Rasmussen and Williams
                Binv = scipy.linalg.inv(B)
                tmp = np.dot(H, self.alpha) + np.dot(Binv, b) # H * K^-1 * y + B^-1 * b

                hl = np.dot(H, self.invL.T)
                M_inv  = Binv + np.dot(hl, hl.T) # here M = (inv(B) +
                                                 # H*K^-1*H.T)^-1 is
                                                 # the posterior
                                                 # covariance matrix
                                                 # on the params.
                self.c = scipy.linalg.cholesky(M_inv, lower=True) # c = sqrt(inv(B) + H*K^-1*H.T)

                self.beta_bar = scipy.linalg.cho_solve((self.c, True), tmp)
                self.invc = scipy.linalg.inv(self.c)
                self.HKinv = np.dot(hl, self.invL)


            # precompute training set log likelihood, so we don't need
            # to keep L around.
            if self.basisfns:
                z = np.dot(H.T, b) - y
            else:
                z = None
                H=None
                B=None

            self._compute_marginal_likelihood(L=L, z=z, B=B, H=H, K=K)

            if compute_grad:
                self.ll_grad = self._log_likelihood_gradient(z=z, K=K, H=H, B=B)



    def _compute_marginal_likelihood(self, L, z=None, B=None, H=None, K=None):
        # to compute log(det(K)), we use the trick that the
        # determinant of a symmetric pos. def. matrix is the
        # product of squares of the diagonal elements of the
        # Cholesky factor
        if not self.basisfns:
            ld2_K = np.log(np.diag(L)).sum()
            self.ll =  -.5 * (np.dot(self.y.T, self.alpha) + self.n * np.log(2*np.pi)) - ld2_K
        else:

            # warning: commented out code (in quotes) is not correct.
            # alternate code below is (I think) correct, but might be slower.


            # here we follow eqn 2.43 in R&W
            #
            # let z = H.T*b - y, then we want
            # .5 * z.T * (K + H.T * B * H)^-1 * z
            # minus some other stuff (dealt with below).
            # by the matrix inv lemma,
            # (K + H.T * B * H)^-1
            # = Kinv - Kinv*H.T*(Binv + H*Kinv*H.T)^-1*H*Kinv
            # = Kinv - Kinv*H.T*     invc.T * invc    *H*Kinv
            # = Kinv - (invc * HKinv)^T (invc * HKinv)
            #
            # so we have z.T * Kinv * z - z.T * (other thing) * z
            # i.e.:            term1    -     term2
            # in the notation of the code.

            tmp1 = np.dot(self.invL, z)
            term1 = np.dot(tmp1.T, tmp1)

            tmp2 = np.dot(self.HKinv, z)
            tmp3 = np.dot(self.invc, tmp2)
            term2 = np.dot(tmp3.T, tmp3)

            # following eqn 2.43 in R&W, we want to compute
            # log det(K + H.T * B * H). using the matrix inversion
            # lemma, we instead compute
            # log det(K) + log det(B) + log det(B^-1 + H*K^-1*H.T)
            ld2_K = np.log(np.diag(L)).sum()
            ld2 =  np.log(np.diag(self.c)).sum() # det( B^-1 - H * K^-1 * H.T )
            ld_B = np.log(np.linalg.det(B))

            # eqn 2.43 in R&W, using the matrix inv lemma
            self.ll = -.5 * (term1 - term2 + self.n * np.log(2*np.pi) + ld_B) - ld2_K - ld2

            """
            # method 2
            sqrt_B = scipy.linalg.cholesky(B, lower=True)
            tmp = np.dot(sqrt_B.T, H)
            K_HBH = K + np.dot(tmp.T, tmp)
            K_HBH_sqrt = scipy.linalg.cholesky(K_HBH, lower=True)
            K_HBH_sqrt_inv = scipy.linalg.inv(K_HBH_sqrt)
            K_HBH_inv = np.dot(K_HBH_sqrt_inv.T, K_HBH_sqrt_inv)

            tmp = np.dot(K_HBH_sqrt_inv, z)
            main_term = np.dot(tmp.T, tmp)

            det_term = np.log(np.diag(K_HBH_sqrt)).sum()

            self.ll = -.5 * main_term - det_term - .5 * self.n * np.log(2*np.pi)
            """

    def get_query_K(self, X1):
        # avoid recomputing the kernel if we're evaluating at the same
        # point multiple times. This is effectively a size-1 cache.
        try:
            self.query_hsh
        except AttributeError:
            self.query_hsh = None

        hsh = hashlib.sha1(X1.view(np.uint8)).hexdigest()
        if hsh != self.query_hsh:
            self.query_K = self.kernel(self.X, X1)
            self.query_hsh = hsh

            if self.basisfns:
                H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=float)
                self.query_R = H - np.dot(self.HKinv, self.query_K)

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.query_K

    def sample(self, X1, include_obs=False):
        """
        Sample from the GP posterior at a set of points given by the rows of X1.

        Default is to sample values of the latent function f. If obs=True, we instead
        sample observed values (i.e. we include observation noise)
        """

        (n,d) = X1.shape
        means = np.reshape(self.predict(X1), (-1, 1))
        K = self.covariance(X1, include_obs=include_obs)
        samples = np.random.randn(n, 1)

        L = scipy.linalg.cholesky(K, lower=True)
        samples = means + np.dot(L, samples)

        return samples

    def predict(self, X1, parametric_only=False):
        """
        Predict the posterior mean, at a set of points given by the rows of X1.
        """

        if parametric_only:
            H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=float)
            gp_pred = np.dot(H.T, self.beta_bar)
        else:
            Kstar = self.get_query_K(X1)
            gp_pred = self.mu + np.dot(Kstar.T, self.alpha)
            if self.basisfns:
                R = self.query_R
                mean_pred = np.dot(R.T, self.beta_bar)
                gp_pred += mean_pred

        return gp_pred

    def covariance(self, X1, include_obs=False, parametric_only=False, pad=1e-8):
        """
        Compute the posterior covariance matrix at a set of points given by the rows of X1.

        Default is to compute the covariance of f, the latent function values. If obs_covar
        is True, we instead compute the covariance of y, the observed values.

        By default, we add a tiny bit of padding to the diagonal to counteract any potential
        loss of positive definiteness from numerical issues. Setting pad=0 disables this.

        """

        Kstar = self.get_query_K(X1)
        tmp = np.dot(self.invL, Kstar)
        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs) - np.dot(tmp.T, tmp)
        else:
            n = X1.shape[0]
            gp_cov = np.zeros((n,n))

        if self.basisfns:
            R = self.query_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)

            gp_cov += mean_cov


        gp_cov += pad * np.eye(gp_cov.shape[0])
        return gp_cov

    def variance(self, X1, **kwargs):
        return np.diag(self.covariance(X1, **kwargs))

    def param_predict(self):
        return self.beta_bar

    def param_covariance(self, chol=False):
        if chol:
            return self.invc
        else:
            return np.dot(self.invc.T, self.invc)

    def param_sample(self, n=1):
        samples = np.random.randn(len(self.beta_bar), n)
        samples = np.reshape(self.beta_bar, (1, -1)) + np.dot(self.invc.T, samples).T
        return samples

    def posterior_log_likelihood(self, X1, y):
        """
        The log probability of the observations (X1, y) under the posterior distribution.
        """

        y = np.array(y)
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        K = self.covariance(X1)
        y = y-self.predict(X1)

        if n==1:
            var = K[0,0]
            ll1 = - .5 * ((y)**2 / var + np.log(2*np.pi*var) )

        L = scipy.linalg.cholesky(K, lower=True)
        ld2 = np.log(np.diag(L)).sum() # this computes .5 * log(det(K))
        alpha = scipy.linalg.cho_solve((L, True), y)
        ll =  -.5 * ( np.dot(y.T, alpha) + n * np.log(2*np.pi)) - ld2
        return ll


    def log_likelihood(self):
        """
        Likelihood of the training data under the prior distribution. (this is primarily determined by the covariance kernel and its hyperparameters)
        """

        return self.ll

    def _log_likelihood_gradient(self, z=None, K=None, H=None, B=None):
        """
        Gradient of the training set log likelihood with respect to the kernel hyperparams.
        """

        n = self.kernel.nparams
        grad = np.zeros((n,))

        if not self.basisfns:

            for i in range(n):
                dKdi = self.kernel.derivative_wrt_i(i, self.X, self.X, identical=True)
                dlldi = .5 * np.dot(self.alpha.T, np.dot(dKdi, self.alpha))

                # here we use the fact:
                # trace(AB) = sum_{ij} A_ij * B_ij
                Kinv = np.dot(self.invL.T, self.invL)
                dlldi -= .5 * np.sum(np.sum(Kinv.T * dKdi))

                grad[i] = dlldi

        else:
            Kinv = np.dot(self.invL.T, self.invL)
            tmp = np.dot(self.invc, self.HKinv)
            K_HBH_inv = Kinv - np.dot(tmp.T, tmp)
            alpha_z = np.dot(K_HBH_inv, z)

            for i in range(n):
                dKdi = self.kernel.derivative_wrt_i(i, self.X, self.X, identical=True)

                dlldi = .5 * np.dot(alpha_z.T, np.dot(dKdi, alpha_z))

                # here we use the fact:
                # trace(AB) = sum_{ij} A_ij * B_ij
                dlldi -= .5 * np.sum(np.sum(K_HBH_inv.T * dKdi))

                grad[i] = dlldi

        return grad


    def pack_npz(self):
        d = {}
        if self.basisfns:
            param_info = {'c': self.c,
                          'beta_bar': self.beta_bar,
                          'invc': self.invc,
                          'HKinv': self.HKinv,
                          'basisfns': np.array([marshal_fn(f) for f in self.basisfns], dtype=object),
                          }
        else:
            param_info = {'basisfns': None}
        d.update(param_info)
        d['X']  = self.X,
        d['y'] =self.y,
        d['mu']  = np.array((self.mu,)),
        d['kernel'] =self.kernel,
        d['alpha'] =self.alpha,
        d['invL'] =self.invL,
        d['ll'] =self.ll,
        return d

    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        d = self.pack_npz()
        with open(filename, 'w') as f:
            np.savez(f, **d)

    def unpack_npz(self, npzfile):
        self.X = npzfile['X'][0]
        self.y = npzfile['y'][0]
        self.mu = npzfile['mu'][0][0]
        self.kernel = npzfile['kernel'].item()
        self.alpha = npzfile['alpha'][0]
        self.invL = npzfile['invL'][0]
        self.ll = npzfile['ll'][0]
        self.basisfns = npzfile['basisfns']
        if self.basisfns is not None and len(self.basisfns.shape) > 0:
            self.beta_bar = npzfile['beta_bar']
            self.c = npzfile['c']
            self.invc = npzfile['invc']
            self.HKinv = npzfile['HKinv']
            self.basisfns = [unmarshal_fn(code) for code in self.basisfns]

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        del npzfile.f
        npzfile.close()

        self.n = self.X.shape[0]
