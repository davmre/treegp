import os
import time
import numpy as np
import collections
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scikits.sparse.cholmod
import pyublas
import hashlib
import types
import marshal


class JointGP():

    def __init__(self):

        self.K = sparse_cov_matrix()
        self.alphas = alphas() # make sure to set this up in row/col order such that the alphas for a single process are contiguous
        self.training_obs_variances = variances()

        self.tree = build_tree()
        self.sparse_threshold = 1e-8
        self.cov_main = cov_main
        self.noise_var = 0.5

    def sparse_kernel(self, X, tree=None, max_distance=None):
        predict_tree = self.tree if tree is None else tree

        if max_distance is None:
            if self.cov_main.wfn_str=="se" and self.sparse_threshold>0:
                max_distance = np.sqrt(-np.log(self.sparse_threshold))
            elif self.cov_main.wfn_str.startswith("compact"):
                max_distance = 1.0
            else:
                max_distance = 1e300

        entries = tree.sparse_training_kernel_matrix(X, max_distance, False)
        spK = scipy.sparse.coo_matrix((entries[:,2], (entries[:,1], entries[:,0])), shape=(self.n, len(X)), dtype=float)
        return spK

    def get_query_K_sparse(self, X1):
        # avoid recomputing the kernel if we're evaluating at the same
        # point multiple times. This is effectively a size-1 cache.
        try:
            self.querysp_hsh
        except AttributeError:
            self.querysp_hsh = None
        hsh = hashlib.sha1(X1.view(np.uint8)).hexdigest()
        if hsh != self.querysp_hsh:
            self.querysp_K = self.sparse_kernel(X1)
            self.querysp_hsh = hsh
        return self.querysp_K

    def predict(self, idx, cond, parametric_only=False):
        pass

    def covariance(self, cond, include_obs=False, pad=1e-8):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        Kstar = self.get_query_K_sparse(X1)
        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:

                f = self.factor(Kstar)
                qf = (Kstar.T * f).todense()

                """
                # alternate form using solve_L
                P = factor.P()
                kp = kstar[P]
                flp = factor.solve_L(kp)
                newdata = flp.data / factor.D()[flp.nonzero()[0]]
                flp2 = flp.copy()
                flp.data = newdata
                qf = (flp2.T * flp).todense()
                """

                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))

        if self.n_features > 0:
            R = self.querysp_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])

        if self.predict_tree_fic is not None:
            gp_cov += np.diag(self.covariance_diag_correction(X1))

        return gp_cov
