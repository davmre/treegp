from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sys
import hashlib
import itertools, pickle, traceback

import numpy as np
import scipy.linalg, scipy.optimize

import kernels

def log_det(X):
    w = np.linalg.eigvalsh(X)
    return np.sum(np.log(w))

class GaussianProcess:


    def __init__(self, X=None, y=None, kernel="se", kernel_params=(1, 1,), kernel_priors = None, kernel_extra = None, K=None, inv=True, mean="constant", fname=None):

        """ Initialize a Gaussian process by providing EITHER the training data (X, y) and kernel info (kernel, kernel_params, kernel_priors, kernel_extra) OR  the filename of a serialized GP.
        'K' is an optional precomputed kernel matrix (for efficiency).
        'mean' specifies a method for estimating the function mean ("zero", "constant", or "linear").
        """

        if fname is not None:
            self.load_trained_model(fname)
        else:
            self.kernel_name = kernel
            self.kernel_params = np.asarray(kernel_params)
            self.kernel_priors = kernel_priors
            self.kernel = kernels.setup_kernel(kernel, kernel_params, kernel_extra, kernel_priors)
            self.X = X
            self.n = X.shape[0]

            if mean == "zero":
                self.mu = 0
                self.y = y
            if mean == "constant":
                self.mu = np.mean(y)
                self.y = y - self.mu
            if mean == "linear":
                raise RuntimeError("linear mean not yet implemented...")
            self.mean = mean

            # train model
            if K is None:
                K = self.kernel(X, X)

            self.L = None
            self.alpha = None
            self.Kinv = None
            try:
                self.L = scipy.linalg.cholesky(K, lower=True)
                self.alpha = scipy.linalg.cho_solve((self.L, True), self.y)
            except np.linalg.linalg.LinAlgError:
                #u,v = np.linalg.eig(K)
                #print K, u
                raise
            except ValueError:
                raise

            # precompute training set log likelihood, so we don't need
            # to keep L around.
            # to compute log(det(K)), we use the trick that the
            # determinant of a symmetric pos. def. matrix is the
            # product of squares of the diagonal elements of the
            # Cholesky factor
            ld2 = np.log(np.diag(self.L)).sum()
            self.ll =  -.5 * (np.dot(self.y.T, self.alpha) + self.n * np.log(2*np.pi)) - ld2

            if inv:
                self.__invert_kernel_matrix()

    def __invert_kernel_matrix(self):
        if self.Kinv is None and self.L is not None:
            invL = scipy.linalg.inv(self.L)
            self.Kinv = np.dot(invL.T, invL)
            self.L = None # no need to keep this around anymore

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
            #print "cache fail: model %d called with " % (len(self.alpha)), X1
        #else:
            #print "cache hit!"
        return self.query_K

    def sample(self, X1):
        """
        Sample from the GP posterior at a set of points given by the rows of X1.
        """

        X1 = np.array(X1)
        if len(X1.shape) == 1:
            X1 = np.reshape(X1, (1, -1))

        (n,d) = X1.shape
        means = self.predict(X1)
        K = self.covariance(X1)

        L = scipy.linalg.cholesky(K, lower=True)
        samples = np.random.randn(n, 1)
        samples = means + np.dot(L, samples)
        return samples

    def predict(self, X1):
        """
        Predict the posterior mean, at a set of points given by the rows of X1.
        """
        K = self.get_query_K(X1)
        return self.mu + np.dot(K.T, self.alpha)

    def covariance(self, X1):
        """
        Compute the posterior covariance matrix at a set of points given by the rows of X1.
        """

        self.__invert_kernel_matrix()
        K = self.get_query_K(X1)
        return self.kernel(X1,X1) - np.dot(K.T, np.dot(self.Kinv, K))

    def variance(self, X1, with_obs_noise=True):
        return np.diag(self.covariance(X1)) + self.kernel_params[0]

    def posterior_log_likelihood(self, X1, y):
        """
        The log probability of the observations (X1, y) under the posterior distribution.
        """

        y = np.array(y)
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        self.__invert_kernel_matrix()

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

    def log_likelihood_gradient(self):
        """
        Gradient of the training set log likelihood with respect to the kernel hyperparams.
        """

        grad = np.zeros(self.kernel_params.shape)

        self.__invert_kernel_matrix()

        for i,p in enumerate(self.kernel_params):

            dKdi = self.kernel.derivative_wrt_i(i, self.X, self.X)
            dlldi = .5 * np.dot(self.alpha.T, np.dot(dKdi, self.alpha))

            # here we use the fact:
            # trace(AB) = sum_{ij} A_ij * B_ij
            dlldi -= .5 * np.sum(np.sum(self.Kinv.T * dKdi))


#            print "dlldi is %f = .5 * %f - .5 * %f" % (dlldi, dlldi1*2, (dlldi1-dlldi)*2)
#            print "norm2 alpha is %f" % (np.dot(self.alpha,  self.alpha))
#            alt = .5 * np.dot(self.y.T, np.dot(self.Kinv, np.dot(dKdi, np.dot(self.Kinv, self.y))))
#            print "alt is %f" % (alt)

            grad[i] = dlldi
 #       print self.kernel_params, "returning grad", grad
        return grad

    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """

        kname = np.array((self.kernel_name,))
        mname = np.array((self.mean,))
        np.savez(filename, X = self.X, y=self.y, mu = np.array((self.mu,)), kernel_name=kname, kernel_params=self.kernel_params, mname = mname, alpha=self.alpha, Kinv=self.Kinv, L=self.L, ll=self.ll)

        #TODO: Use 'marshal' module or inspect.getsource() to serialize the entire kernel including possible outside functions.

    def unpack_npz(self, npzfile):
        self.X = npzfile['X']
        self.y = npzfile['y']
        self.mu = npzfile['mu'][0]
        self.mean = npzfile['mname'][0]
        self.kernel_name = npzfile['kernel_name'][0]
        self.kernel_params = npzfile['kernel_params']
#        self.kernel_extra = npzfile['kernel_extra']
        self.alpha = npzfile['alpha']
        self.Kinv = npzfile['Kinv']
        self.L = npzfile['L']
        self.ll = npzfile['ll']

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        del npzfile.f
        npzfile.close()

        self.n = self.X.shape[0]
        self.kernel = kernels.setup_kernel(self.kernel_name, self.kernel_params, extra=None)
#    def validation_loss(self, trainIdx, valIdx, kernel_params, loss_fn):




def main():
    X = np.array(((0,0), (0,1), (1,0), (1,1)))
    y = np.array((1, 50, 50, 15))


    gp = GaussianProcess(X=X, y=y, kernel="sqexp", kernel_params=(1,), sigma=0.01)
    print gp.predict((0,0))

#    print pickle.dumps(gp)

#    plot_interpolated_surface(gp, X, y)

if __name__ == "__main__":
    main()
