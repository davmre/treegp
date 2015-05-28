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

from features import featurizer_from_string, recover_featurizer
from cover_tree import VectorTree, MatrixTree
from util import mkdir_p

import scipy.weave as weave
from scipy.weave import converters


def marshal_fn(f):
    if f.func_closure is not None:
        raise ValueError("function has non-empty closure %s, cannot marshal!" % repr(f.func_closure))
    s = marshal.dumps(f.func_code)
    return s

def unmarshal_fn(dumped_code):
    f_code = marshal.loads(dumped_code)
    f = types.FunctionType(f_code, globals())
    return f

def sparse_kernel_from_tree(tree, X, sparse_threshold, identical, noise_var):
    max_distance = np.sqrt(-np.log(sparse_threshold)) # assuming a SE kernel
    n = len(X)
    entries = tree.sparse_training_kernel_matrix(X, max_distance, False)
    spK = scipy.sparse.coo_matrix((entries[:,2], (entries[:,0], entries[:,1])), shape=(n,n), dtype=float)

    if identical:
        spK = spK + noise_var * scipy.sparse.eye(spK.shape[0])
    spK = spK + 1e-8 * scipy.sparse.eye(spK.shape[0])
    return spK.tocsc()

def prior_ll(params, priors):
    ll = 0
    for (param, prior) in zip(params, priors):
        if prior is not None:
            ll += prior.log_p(param)
    return ll

def prior_grad(params, priors):
    n = len(params)
    grad = np.zeros((n,))
    for (i, param, prior) in zip(range(n), params, priors):
        if prior is not None:
            grad[i] = prior.deriv_log_p(param)
    return grad

def prior_sample_sparse(X, cov, noise_var, sparse_threshold=1e-20):
    n = X.shape[0]
    predict_tree = VectorTree(X, 1, cov.dfn_str, cov.dfn_params, cov.wfn_str, cov.wfn_params)

    spK = sparse_kernel_from_tree(predict_tree, X, sparse_threshold, True, noise_var)
    factor = scikits.sparse.cholmod.cholesky(spK)
    L = factor.L()
    P = factor.P()
    Pinv = np.argsort(P)
    z = np.random.randn(n)
    y = np.array((L * z)[Pinv]).reshape((-1,))
    return y

def mcov(X, cov, noise_var, X2=None):
    n = X.shape[0]
    predict_tree = VectorTree(X, 1, cov.dfn_str, cov.dfn_params, cov.wfn_str, cov.wfn_params)

    if X2 is None:
        K = predict_tree.kernel_matrix(X, X, False)
        K += np.eye(n) * noise_var
    else:
        K = predict_tree.kernel_matrix(X, X2, False)
    return K

def prior_sample(X, cov, noise_var, sparse_threshold=1e-20, return_K=False):
    n =  X.shape[0]
    K = mcov(X, cov, noise_var)
    L = np.linalg.cholesky(K)

    z = np.random.randn(n)
    y = np.array(np.dot(L, z)).reshape((-1,))

    if return_K:
        return y, K
    else:
        return y


def gaussian_logp(y, K):
    n = K.shape[0]
    L = scipy.linalg.cholesky(K, lower=True)
    ld2 = np.log(np.diag(L)).sum() # this computes .5 * log(det(K))
    alpha = scipy.linalg.cho_solve((L, True), y)
    ll =  -.5 * ( np.dot(y.T, alpha) + n * np.log(2*np.pi)) - ld2

    grad = -scipy.linalg.cho_solve((L, True), y)
    return ll, grad

def ll_under_GPprior(X, y, cov, noise_var, K=None):
    n = X.shape[0]
    predict_tree = VectorTree(X, 1, cov.dfn_str, cov.dfn_params, cov.wfn_str, cov.wfn_params)

    K = predict_tree.kernel_matrix(X, X, False)
    K += np.eye(n) * noise_var

    ll, grad = gaussian_logp(y, K)

    return ll, grad


def dgaussian(r, prec, dcov, dmean=None):
    # derivative of gaussian likelihood wrt derivatives of the mean, cov matrices
    r = r.reshape((-1, 1))

    #dprec = -np.dot(prec, np.dot(dcov, prec))

    dprec_r = -np.dot(prec, np.dot(dcov, np.dot(prec, r)))

    dll_dcov = -.5*np.dot(r.T, dprec_r)
    #dll_dcov -= .5*np.trace(np.dot(prec, dcov))
    dll_dcov -= .5*np.trace(np.dot(prec, dcov))
    dll = dll_dcov

    if dmean is not None:
        dmean = dmean.reshape((-1, 1))
        dll_dmean = np.dot(r.T, np.dot(prec, dmean))
        dll += dll_dmean

    return dll

def dgaussian_rank1(r, alpha, prec, dcov_v, p, dmean=None):
    # derivative of gaussian likelihood wrt a rank-1 update
    # in the cov matrix, where dcov_v is the change in the p'th
    # row and column of the cov matrix (and assume dcov_v[p]=0 for symmetry).
    # Also optionally dmean is a scalar, the change in the p'th entry of the mean vector

    r = r.reshape((-1, 1))


    #t1 = -np.dot(prec, dcov)

    t1 = -np.outer(prec[p,:], dcov_v)
    t1[:, p] = -np.dot(prec, dcov_v)

    dll_dcov = -.5*np.dot(r.T, np.dot(t1, alpha))
    dll_dcov += .5*np.trace(t1)
    dll = dll_dcov

    if dmean is not None:
        dll_dmean = np.dot(r.T, dmean * prec[:, p])
        dll += dll_dmean

    return dll


def unpack_gpcov(d, prefix):
    try:
        wfn_str = d[prefix+'_wfn_str']
        wfn_params = d[prefix+'_wfn_params']
        dfn_str = d[prefix+'_dfn_str']
        dfn_params = d[prefix+'_dfn_params']
        try:
            Xu = d[prefix+'_Xu']
        except KeyError:
            Xu = None
        return GPCov(wfn_params=wfn_params, dfn_params=dfn_params, wfn_str=wfn_str, dfn_str=dfn_str, Xu=Xu)
    except KeyError:
        return None


def sort_morton(X, *args):

    def cmp_zorder(a, b):
            j = 0
            k = 0
            x = 0
            dim = len(a)
            for k in range(dim):
                y = a[k] ^ b[k]
                if less_msb(x, y):
                    j = k
                    x = y
            return a[j] - b[j]

    def less_msb(x, y):
        return x < y and x < (x ^ y)

    Xint = np.array((X + np.min(X, axis=0)) * 10000, dtype=int)
    p = sorted(np.arange(Xint.shape[0]), cmp= lambda i,j : cmp_zorder(Xint[i,:], Xint[j,:]))

    returns = [np.array(X[p,:], copy=True)]
    for y in args:
        returns.append(None if y is None else np.array(y[p], copy=True))

    return tuple(returns)

class GPCov(object):
    def __init__(self, wfn_params, dfn_params,
                 wfn_str="se", dfn_str="euclidean",
                 wfn_priors=None, dfn_priors=None,
                 Xu=None):
        self.wfn_str =str(wfn_str)
        self.wfn_params = np.array(wfn_params, dtype=float)
        self.dfn_str = str(dfn_str)
        self.dfn_params = np.array(dfn_params, dtype=float)
        self.Xu = np.asarray(Xu) if Xu is not None else None

        self.wfn_priors = wfn_priors
        self.dfn_priors = dfn_priors
        if self.wfn_priors is None:
            self.wfn_priors = [None,] * len(self.wfn_params)
        if self.dfn_priors is None:
            self.dfn_priors = [None,] * len(self.dfn_params)

    def copy(self):
        return GPCov(wfn_params = self.wfn_params.copy(), dfn_params=self.dfn_params.copy(),
                     dfn_str = self.dfn_str, wfn_str=self.wfn_str,
                     wfn_priors=self.wfn_priors, dfn_priors=self.dfn_priors, Xu= self.Xu.copy() if self.Xu is not None else None)

    def tree_params(self):
        return (self.dfn_str, self.dfn_params, self.wfn_str, self.wfn_params)

    def packed(self, prefix):
        return {prefix + "_wfn_str": self.wfn_str,
                prefix + "_wfn_params": self.wfn_params,
                prefix + "_dfn_str": self.dfn_str,
                prefix + "_dfn_params": self.dfn_params,
                prefix + "_Xu": self.Xu}

    def bounds(self, include_xu=True):
        b = [(1e-8,None),] * (len(self.wfn_params) + len(self.dfn_params))
        if include_xu and self.Xu is not None:
            b += [(None, None),] * self.Xu.size
        return b

    def flatten(self, include_xu=True):
        v = np.concatenate([self.wfn_params, self.dfn_params])
        if include_xu and self.Xu is not None:
            v = np.concatenate([v, self.Xu.flatten()])
        return v

    def prior_logp(self):
        return prior_ll(self.dfn_params, self.dfn_priors) + prior_ll(self.wfn_params, self.wfn_priors)

    def prior_grad(self, include_xu=True):
        v = np.concatenate([prior_grad(self.wfn_params, self.wfn_priors) ,
                            prior_grad(self.dfn_params, self.dfn_priors)])
        if include_xu and self.Xu is not None:
            v = np.concatenate([v, np.zeros((self.Xu.size,))])
        return v

    def __repr__(self):
        s = self.wfn_str + str(self.wfn_params) + ", " + self.dfn_str + str(self.dfn_params)
        return s


class GP(object):

    def standardize_input_array(self, c, **kwargs):
        assert(len(c.shape) == 2)
        return c

    def training_kernel_matrix(self, X):
        K = self.kernel(X, X, identical=True)

        d = np.diag(K).copy() #+ 1e-8
        if self.y_obs_variances is not None:
            d += self.y_obs_variances
        np.fill_diagonal(K, d)

        return K

    def sparse_training_kernel_matrix(self, X):
        K = self.sparse_kernel(X, identical=True)
        d = K.diagonal().copy() # + 1e-8
        if self.y_obs_variances is not None:
            d += self.y_obs_variances
        K.setdiag(d)

        return K.tocsc()

    def invert_kernel_matrix(self, K):
        alpha = None
        t0 = time.time()
        L = scipy.linalg.cholesky(K, lower=True) # c = sqrt(inv(B) + H*K^-1*H.T)
        factor = lambda z : scipy.linalg.cho_solve((L, True), z)
        t1 = time.time()
        self.timings['chol_factor'] = t1-t0

        alpha = factor(self.y)
        t2 = time.time()
        self.timings['solve_alpha'] = t2-t1

        Kinv = np.linalg.inv(K)
        t3 = time.time()
        self.timings['solve_Kinv'] = t3-t2

        I = np.dot(Kinv[0,:], K[:,0])
        if np.abs(I - 1) > 0.01:
            print "WARNING: poorly conditioned inverse (I=%f)" % I

        return alpha, factor, L, Kinv

    def sparse_invert_kernel_matrix(self, K):
        alpha = None
        t0 = time.time()
        factor = scikits.sparse.cholmod.cholesky(K)
        t1 = time.time()
        self.timings['chol_factor'] = t1-t0

        alpha = factor(self.y)
        t2 = time.time()
        self.timings['solve_alpha'] = t2-t1
        Kinv = factor(scipy.sparse.eye(K.shape[0]).tocsc())
        t3 = time.time()
        self.timings['solve_Kinv'] = t3-t2

        I = (Kinv.getrow(0) * K.getcol(0)).todense()[0,0]
        if np.abs(I - 1) > 0.01:
            print "WARNING: poorly conditioned inverse (I=%f)" % I

        unpermuted_L = factor.L()
        P = factor.P()
        Pinv = np.argsort(P)
        L = unpermuted_L[Pinv,:][:,Pinv]

        return alpha, factor, L, Kinv


    def sparsify(self, M):
        import scipy.sparse
        if scipy.sparse.issparse(M):
            M = M.copy()
            chunksize=1000000
            nchunks = len(M.data)/chunksize+1
            for i in range(nchunks):
                cond = (np.abs(M.data[i*chunksize:(i+1)*chunksize]) < self.sparse_threshold)
                M.data[i*chunksize:(i+1)*chunksize][cond] = 0
            M.eliminate_zeros()
            return M
        else:
            return scipy.sparse.csc_matrix(np.asarray(M) * np.asarray(np.abs(M) > self.sparse_threshold))

    def sort_events(self, X, y):
        combined = np.hstack([X, np.reshape(y, (-1, 1))])
        combined_sorted = np.array(sorted(combined, key = lambda x: x[0]), dtype=float)
        X_sorted = np.array(combined_sorted[:, :-1], copy=True, dtype=float)
        y_sorted = combined_sorted[:, -1].flatten()
        return X_sorted, y_sorted


    ##############################################################################
    # methods for building / interacting with low-rank additive covariances

    def get_data_features(self, X):
        # compute the full set of features for a matrix X of test points
        features = np.zeros((self.n_features, X.shape[0]))

        i = 0
        if self.featurizer is not None:
            F = self.featurizer(X)
            i = F.shape[0]
            features[:i,:] = F

        if self.predict_tree_fic is not None:
            features[i:,:] = self.kernel(self.cov_fic.Xu, X, predict_tree=self.predict_tree_fic)

        return features


    def combine_lowrank_models(self, H, b, B, K_fic_un, K_fic_uu):
        # for training lowrank models: given feature representations
        # (H, K_fic_un) and prior covariances (B, K_fic_uu) for two
        # additive models, return the combined feature representation,
        # combined parameter prior mean (assuming the FIC parameters
        # have mean 0) and combined parameter precision matrix (we
        # assume the parameters of the two models are independent, so
        # this is block diagonal).

        N = 0
        self.n_features = 0
        self.n_param_features = 0
        if H is not None:
            self.n_param_features = H.shape[0]
            self.n_features += self.n_param_features
            N = H.shape[1]
        if K_fic_un is not None:
            self.n_features += K_fic_un.shape[0]
            N = K_fic_un.shape[1]

        features = np.zeros((self.n_features, N))
        mean = np.zeros((self.n_features,))
        cov_inv = np.zeros((self.n_features, self.n_features))

        if H is not None:
            features[0:self.n_param_features, :] = H
            mean[0:self.n_param_features] = b
            cov_inv[0:self.n_param_features,0:self.n_param_features] = np.linalg.inv(B)
        if K_fic_un is not None:
            features[self.n_param_features:, :] = K_fic_un
            cov_inv[self.n_param_features:,self.n_param_features:] = K_fic_uu

        return features, mean, cov_inv

    def build_low_rank_model(self, alpha, Kinv_sp, H, b, Binv):
        """
        let n be the training size; we'll use an additional rank-m approximation.
        the notation here follows section 2.7 in Rasmussen & Williams. For
        simplicity, K refers to the observation covariance matrix rather than the
        underlying function covariance (i.e. it might really be K+noise_var*I, or
        K+diag(y_obs_variances), etc.)

        takes:
         alpha: n x 1, equal to K^-1 y
         Kinv_sp: n x n sparse matrix, equal to K^-1
         H: n x m features of training data (this is Qfu for FIC)
         b: m x 1 prior mean on feature weights (this is 0 for FIC)
         B: m x m prior covariance on feature weights (this is Quu for FIC)

        returns:
         invc = inv(chol(M)), where M = (B^-1 + H K^-1 H^T)^-1 is the
                posterior covariance matrix on feature weights
         beta_bar = M (HK^-1y + B^-1 b) gives the weights for the correction
                    of the low-rank component to the mean prediction
         HKinv = HK^-1 comes up in the marginal likelihood computation, so we
                 go ahead and remember the value we compute now.
        """

        # tmp = H * K^-1 * y + B^-1 * b
        tmp = np.reshape(np.asarray(np.dot(H, alpha)), (-1,))
        if b is not None:
            tmp += np.dot(Binv, b)

        HKinv = H * Kinv_sp
        M_inv  = Binv + np.dot(HKinv, H.T)
        c = scipy.linalg.cholesky(M_inv, lower=True)
        beta_bar = scipy.linalg.cho_solve((c, True), tmp)
        invc = scipy.linalg.inv(c)
        return c, invc, beta_bar, HKinv


    def init_csfic_kernel(self, K_cs):
        K_fic_uu = self.kernel(self.cov_fic.Xu, self.cov_fic.Xu, identical=False, predict_tree = self.predict_tree_fic)
        Luu  = scipy.linalg.cholesky(K_fic_uu, lower=True)
        self.Luu = Luu
        K_fic_un = self.kernel(self.cov_fic.Xu, self.X, identical=False, predict_tree = self.predict_tree_fic)

        dc = self.covariance_diag_correction(self.X)
        diag_correction = scipy.sparse.dia_matrix((dc, 0), shape=K_cs.shape)

        K_cs = K_cs + diag_correction
        return K_cs, K_fic_uu, K_fic_un


    def setup_parametric_featurizer(self, X, featurizer_recovery, basis, extract_dim):
        # setup featurizer for parametric mean function, if applicable
        H = None
        self.featurizer = None
        self.featurizer_recovery = None
        if featurizer_recovery is None:
            if basis is not None:
                H, self.featurizer, self.featurizer_recovery = featurizer_from_string(X, basis, extract_dim=extract_dim, transpose=True)
        else:
            self.featurizer, self.featurizer_recovery = recover_featurizer(basis, featurizer_recovery, transpose=True)
            H = self.featurizer(X)
        return H

    ###################################################################################



    def __init__(self, X=None, y=None,
                 y_obs_variances=None,
                 fname=None,
                 noise_var=1.0,
                 cov_main=None,
                 cov_fic=None,
                 basis = None,
                 extract_dim = None,
                 param_mean=None,
                 param_cov=None,
                 featurizer_recovery=None,
                 compute_ll=False,
                 compute_grad=False,
                 compute_xu_grad=False,
                 sparse_threshold=1e-10,
                 K = None,
                 sort_events=True,
                 build_tree=False,
                 compile_tree=None,
                 sparse_invert=True,
                 center_mean=False,
                 ymean=0.0,
                 leaf_bin_width = 0,
                 build_dense_Kinv_hack=False): # WARNING: bin sizes > 0 currently lead to memory leaks


        self.double_tree = None
        if fname is not None:
            self.load_trained_model(fname, build_tree=build_tree, leaf_bin_width=leaf_bin_width, build_dense_Kinv_hack=build_dense_Kinv_hack, compile_tree=compile_tree)
        else:
            if sort_events:
                X, y, y_obs_variances = sort_morton(X, y, y_obs_variances) # arrange events by
                # lon/lat, as a
                # heuristic to expose
                # block structure in the
                # kernel matrix


            self.cov_main, self.cov_fic, self.noise_var, self.sparse_threshold, self.basis = cov_main, cov_fic, noise_var, sparse_threshold, basis

            self.timings = dict()

            if X is not None:
                self.X = np.matrix(X, dtype=float)
                self.y = np.array(y, dtype=float)
                if center_mean:
                    self.ymean = np.mean(y)
                    self.y -= self.ymean
                else:
                    self.ymean = ymean
                    self.y -= self.ymean
                self.n = X.shape[0]
                if y_obs_variances is not None:
                    self.y_obs_variances = np.array(y_obs_variances, dtype=float).flatten()
                else:
                    self.y_obs_variances = None

            else:
                self.X = np.reshape(np.array(()), (0,0))
                self.y = np.reshape(np.array(()), (0,))
                self.n = 0
                self.ymean = ymean
                self.K = np.reshape(np.array(()), (0,0))
                self.Kinv = np.reshape(np.array(()), (0,0))
                self.alpha_r = self.y
                self.ll = np.float('-inf')
                return

            self.predict_tree, self.predict_tree_fic = self.build_initial_single_trees(build_single_trees=sparse_invert)

            # compute sparse training kernel matrix (including per-observation noise if appropriate)
            if sparse_invert:
                self._set_max_distance()
                self.K = self.sparse_training_kernel_matrix(self.X)
            else:
                self.K = self.training_kernel_matrix(self.X)

            # setup the parameteric features, if applicable, and return the feature representation of X
            H = self.setup_parametric_featurizer(X, featurizer_recovery, basis, extract_dim)

            if cov_fic is not None:
                self.K, self.K_fic_uu, self.K_fic_un = self.init_csfic_kernel(self.K)
            else:
                self.K_fic_uu, self.K_fic_un = None, None

            #print "built kernel matrix"

            # invert kernel matrix
            if sparse_invert:
                if type(self.K) == np.ndarray or type(self.K) == np.matrix:
                    self.K = self.sparsify(self.K)
                alpha, self.factor, L, Kinv = self.sparse_invert_kernel_matrix(self.K)
                self.Kinv = self.sparsify(Kinv)
                #print "Kinv is ", len(self.Kinv.nonzero()[0]) / float(self.Kinv.shape[0]**2), "full (vs diag at", 1.0/self.Kinv.shape[0], ")"
            else:
                alpha, self.factor, L, Kinv = self.invert_kernel_matrix(self.K)
                if build_tree:
                    self.Kinv = self.sparsify(Kinv)
                    #print "Kinv is ", len(self.Kinv.nonzero()[0]) / float(self.Kinv.shape[0]**2), "full (vs diag at", 1.0/self.Kinv.shape[0], ")"
                else:
                    self.Kinv=np.matrix(Kinv)

            #print "inverted kernel matrix"

            # if we have any additive low-rank covariances, compute the appropriate terms
            if H is not None or cov_fic is not None:
                HH, b, Binv = self.combine_lowrank_models(H, param_mean, param_cov, self.K_fic_un, self.K_fic_uu)
                self.c, self.invc,self.beta_bar, self.HKinv = self.build_low_rank_model(alpha,
                                                                                        self.Kinv,
                                                                                        HH,
                                                                                        b,
                                                                                        Binv)
                r = self.y - np.dot(HH.T, self.beta_bar)
                self.alpha_r = np.reshape(np.asarray(self.factor(r)), (-1,))

                z = np.dot(HH.T, b) - self.y
            else:
                self.n_features = 0
                self.HKinv = None
                self.alpha_r = np.reshape(alpha, (-1,))
                r = self.y
                z = self.y
                Binv = None
                HH = None

            if build_tree:
                self.build_point_tree(HKinv = self.HKinv, Kinv=self.Kinv, alpha_r = self.alpha_r, leaf_bin_width=leaf_bin_width, build_dense_Kinv_hack=build_dense_Kinv_hack, compile_tree=compile_tree)

            # precompute training set log likelihood, so we don't need
            # to keep L around.
            if compute_ll:
                self._compute_marginal_likelihood(L=L, z=z, Binv=Binv, H=HH, K=self.K, Kinv=self.Kinv)
            else:
                self.ll = -np.inf
            if compute_grad:
                self.ll_grad = self._log_likelihood_gradient(z=z, Kinv=self.Kinv, include_xu=compute_xu_grad)

    def build_initial_single_trees(self, build_single_trees = False):
        if build_single_trees:
            tree_X = pyublas.why_not(self.X)
        else:
            tree_X = np.array([[0.0,] * self.X.shape[1],], dtype=float)

        if self.cov_main is not None:
            predict_tree = VectorTree(tree_X, 1, *self.cov_main.tree_params())
        else:
            predict_tree = None

        if self.cov_fic is not None:
            predict_tree_fic = VectorTree(tree_X, 1, *self.cov_fic.tree_params())
        else:
            predict_tree_fic = None
        return predict_tree, predict_tree_fic

    def _set_max_distance(self):
        if self.cov_main.wfn_str=="se" and self.sparse_threshold>0:
            self.max_distance = np.sqrt(-np.log(self.sparse_threshold))
        elif self.cov_main.wfn_str.startswith("compact"):
            self.max_distance = 1.0
        else:
            self.max_distance = 1e300


    def build_point_tree(self, HKinv, Kinv, alpha_r, leaf_bin_width, build_dense_Kinv_hack=False, compile_tree=None):
        if self.n == 0: return

        self._set_max_distance()

        fullness = len(self.Kinv.nonzero()[0]) / float(self.Kinv.shape[0]**2)
        # print "Kinv is %.1f%% full." % (fullness * 100)
        #if fullness > .15:
        #    raise Exception("not building tree, Kinv is too full!" )

        self.predict_tree.set_v(0, alpha_r.astype(np.float))

        if HKinv is not None:
            self.cov_tree = VectorTree(self.X, self.n_features, *self.cov_main.tree_params())
            HKinv = HKinv.astype(np.float)
            for i in range(self.n_features):
                self.cov_tree.set_v(i, HKinv[i, :])


        nzr, nzc = Kinv.nonzero()
        vals = np.reshape(np.asarray(Kinv[nzr, nzc]), (-1,))

        if build_dense_Kinv_hack:
            self.predict_tree.set_Kinv_for_dense_hack(nzr, nzc, vals)

        t0 = time.time()
        self.double_tree = MatrixTree(self.X, nzr, nzc, *self.cov_main.tree_params())
        self.double_tree.set_m_sparse(nzr, nzc, vals)
        if leaf_bin_width > 0:
            self.double_tree.collapse_leaf_bins(leaf_bin_width)
        t1 = time.time()
        print "built product tree on %d points in %.3fs" % (self.n, t1-t0)
        if compile_tree is not None:
            #source_fname = compile_tree + ".cc"
            #obj_fname = compile_tree + ".o"
            linked_fname = compile_tree + "/main.so"

            if not os.path.exists(linked_fname):
                mkdir_p(compile_tree)
                self.double_tree.compile(compile_tree, 0)
                print "generated source files in ", compile_tree
                import sys; sys.exit(1)

                objfiles = []
                for srcfile in os.listdir(compile_tree):
                    if srcfile.endswith(".cc"):
                        objfile = os.path.join(compile_tree, srcfile[:-3] + ".o")
                        objfiles.append(objfile)
                        os.system("gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -fPIC -I/home/dmoore/.virtualenvs/treegp/local/lib/python2.7/site-packages/pyublas/include -I/home/dmoore/.virtualenvs/treegp/local/lib/python2.7/site-packages/numpy/core/include -I/home/dmoore/local/include/ -I/usr/include/python2.7 -c %s -o %s -O3" % (os.path.join(compile_tree, srcfile), objfile))
                os.system("g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro %s -L/ -lboost_python -o %s" % (" ".join(objfiles), linked_fname))

            import imp
            self.compiled_tree = imp.load_dynamic("compiled_tree", linked_fname)
            self.compiled_tree.init_distance_caches()

    def predict(self, cond, parametric_only=False, eps=1e-8):
        if not self.double_tree: return self.predict_naive(cond, parametric_only)
        X1 = self.standardize_input_array(cond).astype(np.float)

        if parametric_only:
            gp_pred = np.zeros((X1.shape[0],))
        else:
            gp_pred = np.array([self.predict_tree.weighted_sum(0, np.reshape(x, (1,-1)), eps) for x in X1])

        if self.n_features > 0:
            H = self.get_data_features(X1)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        gp_pred += self.ymean
        return gp_pred

    def predict_naive(self, cond, parametric_only=False, eps=1e-8):
        X1 = self.standardize_input_array(cond).astype(np.float)

        if parametric_only:
            gp_pred = np.zeros((X1.shape[0],))
        else:
            Kstar = self.kernel(self.X, X1)
            gp_pred = np.dot(Kstar.T, self.alpha_r)

        if self.n_features > 0:
            H = self.get_data_features(X1)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred


        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        gp_pred += self.ymean

        return gp_pred

    def dKdi(self, X1, X2, i, identical=False):
        if (i == 0):
            dKdi = np.eye(X1.shape[0]) if identical else np.zeros((X1.shape[0], X2.shape[0]))
        elif (i == 1):
            if (len(self.cov_main.wfn_params) != 1):
                raise ValueError('gradient computation currently assumes just a single scaling parameter for weight function, but currently wfn_params=%s' % self.cov_main.wfn_params)
            dKdi = self.kernel(X1, X2, identical=False) / self.cov_main.wfn_params[0]
        else:
            dc = self.predict_tree.kernel_matrix(X1, X2, True)
            dKdi = self.predict_tree.kernel_deriv_wrt_i(X1, X2, i-2, 1 if identical else 0, dc)
        return dKdi

    def dKdx(self, X1, p, i, X2=None, return_vec=False):
        # derivative of kernel(X1, X2) wrt i'th coordinate of p'th point in X1.
        if X2 is None:
            # consider k(X1, X1)
            """
            n = X1.shape[0]
            xp =  np.array(X1[p:p+1,:], copy=True)
            dK = np.zeros((n, n))
            kp = self.predict_tree_fic.kernel_deriv_wrt_xi(xp, X1, 0, i)
            dK[p,:] = kp
            dK[:,p] = kp
            """
            if return_vec:
                dKv = np.zeros((X1.shape[0]))
                self.predict_tree.kernel_deriv_wrt_xi_row(X1, p, i, dKv)
                dKv[p] = 0
                return dKv
            else:
                dK = self.predict_tree.kernel_deriv_wrt_xi(X1, X1, p, i)
                dK[p,p] = 0
                dK = dK + dK.T
        else:
            dK = self.predict_tree.kernel_deriv_wrt_xi(X1, X2, p, i)
        return dK

    def grad_prediction(self, cond, i):
        """
        Compute the derivative of the predictive distribution (mean and
        cov matrix) at a given location with respect to a kernel hyperparameter.
        """
        # ONLY WORKS for zero-mean dense GPs with no inducing points or parametric components
        X1 = self.standardize_input_array(cond).astype(np.float)

        n_main_params = len(self.cov_main.flatten())
        nparams = 1 + n_main_params

        Kstar = self.kernel(X1, self.X)

        dKstar = self.dKdi(X1, self.X, i, identical=False)
        dKy = self.dKdi(self.X, self.X, i, identical=True)

        dKyinv = -np.dot(self.Kinv, np.dot(dKy, self.Kinv))

        Kstar_Kyinv = np.dot(Kstar, self.Kinv)
        d_Kstar_Kyinv = np.dot(Kstar, dKyinv) + np.dot(dKstar, self.Kinv)
        dmean = np.dot(d_Kstar_Kyinv, self.y)

        #Kss = self.kernel(X1, X1)
        dKss = self.dKdi(X1, X1, i, identical=True)

        dqf = np.dot(d_Kstar_Kyinv, Kstar.T) + np.dot(Kstar_Kyinv, dKstar.T)
        dcov = dKss - dqf

        return dmean, dcov

    def grad_prediction_wrt_source_x(self, cond, p, i):
        # ONLY WORKS for zero-mean dense GPs with no inducing points or parametric components
        X1 = self.standardize_input_array(cond).astype(np.float)
        n_main_params = len(self.cov_main.flatten())
        nparams = 1 + n_main_params

        # can share Kstar over different vals of p, i
        Kstar = self.kernel(X1, self.X)
        Kstar_Kyinv = np.dot(Kstar, self.Kinv)

        dKstar = self.dKdx(self.X, p, i, X1).T

        # does not depend on X1: can be precomputed
        # also lots of computation can be shared because only one row(/col) of dKy and dKstar should be nonzero
        dKy = self.dKdx(self.X, p, i)
        dKyinv = -np.dot(self.Kinv, np.dot(dKy, self.Kinv))

        d_Kstar_Kyinv = np.dot(Kstar, dKyinv) + np.dot(dKstar, self.Kinv)

        dm = np.dot(d_Kstar_Kyinv, self.y)
        dqf = np.dot(d_Kstar_Kyinv, Kstar.T) + np.dot(Kstar_Kyinv, dKstar.T)
        dc =  -dqf

        return dm, dc

    def grad_prediction_wrt_target_x(self, cond, p, i):
        # ONLY WORKS for zero-mean dense GPs with no inducing points or parametric components
        X1 = self.standardize_input_array(cond).astype(np.float)

        n_main_params = len(self.cov_main.flatten())
        nparams = 1 + n_main_params

        # can be shared over p, i
        Kstar = self.kernel(X1, self.X)
        Kss = self.kernel(X1, X1, identical=True)

        # both of these are rank-1
        dKss = self.dKdx(X1, p, i)
        dKstar = self.dKdx(X1, p, i, X2=self.X)



        dm = np.dot(dKstar, self.alpha_r)
        dqf = np.dot(dKstar, np.dot(self.Kinv, Kstar.T))
        dc = dKss - dqf - dqf.T

        return dm, dc

    def grad_ll_wrt_X(self):
        # ONLY WORKS for zero-mean dense GPs with no inducing points or parametric components
        n, d = self.X.shape
        llgrad = np.zeros((n, d))

        for p in range(n):
            for i in range(d):

                #dc = self.dKdx(self.X, p, i)
                #llgrad2 = dgaussian(self.y, self.Kinv, dc)

                dcv = self.dKdx(self.X, p, i, return_vec=True)
                llgrad[p,i] = dgaussian_rank1(self.y, self.alpha_r, self.Kinv, dcv, p)

        return llgrad


    def kernel(self, X1, X2, identical=False, predict_tree=None):
        predict_tree = self.predict_tree if predict_tree is None else predict_tree
        K = predict_tree.kernel_matrix(X1, X2, False)
        if identical:
            K += self.noise_var * np.eye(K.shape[0])
        return K

    def sparse_kernel(self, X, identical=False, predict_tree=None, max_distance=None):
        predict_tree = self.predict_tree if predict_tree is None else predict_tree

        if max_distance is None:
            max_distance = self.max_distance

        entries = predict_tree.sparse_training_kernel_matrix(X, max_distance, False)
        spK = scipy.sparse.coo_matrix((entries[:,2], (entries[:,1], entries[:,0])), shape=(self.n, len(X)), dtype=float)

        if identical:
            spK = spK + self.noise_var * scipy.sparse.eye(spK.shape[0])

        return spK

    def get_query_K_sparse(self, X1, no_R=False):
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

            if self.n_features > 0 and not no_R:
                H = self.get_data_features(X1)
                self.querysp_R = H - np.asmatrix(self.HKinv) * self.querysp_K

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.querysp_K

    def get_query_K(self, X1, no_R=False):
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

            if self.n_features > 0 and not no_R:
                H = self.get_data_features(X1)
                self.query_R = H - np.asmatrix(self.HKinv) * self.query_K

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.query_K


    def covariance_diag_correction(self, X):
        K_fic_un = self.kernel(self.cov_fic.Xu, X, identical=False, predict_tree = self.predict_tree_fic)
        B = scipy.linalg.solve(self.Luu, K_fic_un)
        Qvff = np.sum(B*B, axis=0)
        return self.cov_fic.wfn_params[0] - Qvff

    def covariance_spkernel(self, cond, include_obs=False, parametric_only=False, pad=1e-8):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        t1 = time.time()
        Kstar = self.get_query_K_sparse(X1, no_R=True)
        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:
                tmp = self.Kinv * Kstar
                qf = (Kstar.T * tmp).todense()
                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))
        t2 = time.time()
        self.qf_time = t2-t1

        if self.n_features > 0:
            t1 = time.time()
            H = self.get_data_features(X1)
            R = H - np.asmatrix(self.HKinv) * self.querysp_K

            #R = self.querysp_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov
            t2 = time.time()
            self.nonqf_time = t2-t1

        gp_cov += pad * np.eye(gp_cov.shape[0])
        if self.predict_tree_fic is not None:
            gp_cov += np.diag(self.covariance_diag_correction(X1))


        return gp_cov

    def covariance_spkernel_solve(self, cond, include_obs=False, parametric_only=False, pad=1e-8):
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


    def covariance_treedense(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps_abs=1e-8, qf_only=False):

        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]



        if not parametric_only:
            gp_cov = self.kernel(X1, X1, identical=include_obs)

            t1 = time.time()
            if self.n > 0:
                qf = np.zeros(gp_cov.shape)
                for i in range(m):
                    for j in range(m):
                        qf = self.predict_tree.quadratic_form_from_dense_hack(X1[i:i+1], X1[j:j+1], self.max_distance)
                if qf_only:
                    return qf
                gp_cov -= qf
            t2 = time.time()
            self.qf_time = t2-t1


            self.qf_dfn_evals = self.predict_tree.dense_hack_dfn_evals
            self.qf_wfn_evals = self.predict_tree.dense_hack_wfn_evals
            self.qf_terms = self.predict_tree.dense_hack_terms


        else:
            gp_cov = np.zeros((m,m))

        if self.n_features > 0:
            t1 = time.time()
            H = self.get_data_features(X1)
            #H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=np.float64)
            HKinvKstar = np.zeros((self.n_features, m))

            for i in range(self.n_features):
                for j in range(m):
                    HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps_abs)
            R = H - HKinvKstar
            v = np.dot(self.invc, R)
            mc = np.dot(v.T, v)
            gp_cov += mc
            t2 = time.time()
            self.nonqf_time = t2-t1
        else:
            self.nonqf_time = 0


        gp_cov += pad * np.eye(m)

        if self.predict_tree_fic is not None:
            gp_cov += np.diag(self.covariance_diag_correction(X1))

        return gp_cov



    def covariance(self, cond, include_obs=False, parametric_only=False, pad=1e-8, qf_only=False):
        """
        Compute the posterior covariance matrix at a set of points given by the rows of X1.

        Default is to compute the covariance of f, the latent function values. If obs_covar
        is True, we instead compute the covariance of y, the observed values.

        By default, we add a tiny bit of padding to the diagonal to counteract any potential
        loss of positive definiteness from numerical issues. Setting pad=0 disables this.

        """
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        t1 = time.time()
        Kstar = self.get_query_K(X1, no_R=True)
        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:
                tmp = self.Kinv * Kstar
                qf = np.dot(Kstar.T, tmp)
                if qf_only:
                    return qf
                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))
        t2 = time.time()
        self.qf_time = t2-t1


        if self.n_features > 0:
            t1 = time.time()
            H = self.get_data_features(X1)
            R = H - np.asmatrix(self.HKinv) * self.query_K

            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov
            t2 = time.time()
            self.nonqf_time = t2-t1
        else:
            self.nonqf_time = 0

        gp_cov += pad * np.eye(gp_cov.shape[0])

        if self.predict_tree_fic is not None:
            gp_cov += np.diag(self.covariance_diag_correction(X1))


        return gp_cov

    def covariance_double_tree(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps=-1, eps_abs=1e-4, cutoff_rule=1, qf_only=False):

        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]
        cutoff_rule = int(cutoff_rule)

        if not parametric_only:
            gp_cov = self.kernel(X1, X1, identical=include_obs)

            t1 = time.time()
            if self.n > 0:
                qf = np.zeros(gp_cov.shape)
                for i in range(m):
                    for j in range(m):
                        qf[i,j] = self.double_tree.quadratic_form(X1[i:i+1], X1[j:j+1], eps, eps_abs, cutoff_rule)
                if qf_only:
                    return qf
                gp_cov -= qf
            t2 = time.time()
            self.qf_time = t2-t1

            self.qf_terms = self.double_tree.terms
            self.qf_zeroterms = self.double_tree.zeroterms
            self.qf_nodes_touched = self.double_tree.nodes_touched
            self.qf_dfn_evals = self.double_tree.dfn_evals
            self.qf_dfn_misses = self.double_tree.dfn_misses
            self.qf_wfn_evals = self.double_tree.wfn_evals
            self.qf_wfn_misses = self.double_tree.wfn_misses


        else:
            gp_cov = np.zeros((m,m))

        if self.n_features > 0:
            t1 = time.time()
            H = self.get_data_features(X1)
            #H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=np.float64)
            HKinvKstar = np.zeros((self.n_features, m))

            for i in range(self.n_features):
                for j in range(m):
                    HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps_abs)
            R = H - HKinvKstar
            v = np.dot(self.invc, R)
            mc = np.dot(v.T, v)
            gp_cov += mc
            t2 = time.time()
            self.nonqf_time = t2-t1
        else:
            self.nonqf_time = 0

        gp_cov += pad * np.eye(m)

        if self.predict_tree_fic is not None:
            gp_cov += np.diag(self.covariance_diag_correction(X1))



        return gp_cov


    def covariance_compiled(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps_abs=1e-4,  qf_only=False):

        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        if not parametric_only:
            gp_cov = self.kernel(X1, X1, identical=include_obs)

            t1 = time.time()
            if self.n > 0:
                qf = np.zeros(gp_cov.shape)
                for i in range(m):
                    for j in range(m):
                        qf[i,j] = self.compiled_tree.quadratic_form_symmetric(X1[i:i+1], eps_abs)
                if qf_only:
                    return qf
                gp_cov -= qf
            t2 = time.time()
            self.qf_time = t2-t1

            #self.qf_terms = self.compiled_tree.get_terms()
            #self.qf_zeroterms = self.compiled_tree.get_zeroterms()
            #self.qf_nodes_touched = self.compiled_tree.get_nodes_touched()
            #self.qf_dfn_evals = self.compiled_tree.get_dfn_evals()

        else:
            gp_cov = np.zeros((m,m))

        if self.n_features > 0:
            t1 = time.time()
            H = self.get_data_features(X1)
            #H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=np.float64)
            HKinvKstar = np.zeros((self.n_features, m))

            for i in range(self.n_features):
                for j in range(m):
                    HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps_abs)
            R = H - HKinvKstar
            v = np.dot(self.invc, R)
            mc = np.dot(v.T, v)
            gp_cov += mc
            t2 = time.time()
            self.nonqf_time = t2-t1
        else:
            self.nonqf_time = 0

        gp_cov += pad * np.eye(m)

        if self.predict_tree_fic is not None:
            gp_cov += np.diag(self.covariance_diag_correction(X1))



        return gp_cov

    def variance(self,cond, **kwargs):
        v = np.diag(self.covariance(cond, **kwargs))
        return v

    def sample(self, cond, include_obs=True, method="naive"):
        """
        Sample from the GP posterior at a set of points given by the rows of X1.

        Default is to sample observed values (i.e. we include observation noise). If obs=False, we instead
        sample values of the latent function f.
        """

        X1 = self.standardize_input_array(cond)
        (n,d) = X1.shape
        means = np.reshape(self.predict(X1), (-1, 1))


        if method == "naive":
            K = self.covariance(X1, include_obs=include_obs)
        elif method == "sparse":
            K = self.covariance_spkernel(X1, include_obs=include_obs)
        elif method == "tree":
            K = self.covariance_double_tree(X1, include_obs=include_obs)
        else:
            raise Exception("unknown covariance method %s" % method)

        samples = np.random.randn(n, 1)

        L = scipy.linalg.cholesky(K, lower=True)
        samples = means + np.dot(L, samples)


        if len(samples) == 1:
            samples = samples[0]

        return samples

    def param_mean(self):
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

    def deriv_log_p(self, x, cond=None,lp0=None, eps=1e-4, include_obs=True, **kwargs):

        X1 = self.standardize_input_array(cond, **kwargs)
        y = x if isinstance(x, collections.Iterable) else (x,)

        y = np.array(y);
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        K = self.covariance(X1, include_obs=True)
        y = y-self.predict(X1)


        L =  scipy.linalg.cholesky(K, lower=True)
        return -scipy.linalg.cho_solve((L, True), y)

        # X1: kx6 array w/ station and event locations
        # y: k-dimensional vector
        # ignore idx, cond_key, cond_idx


        # return k-dimensional vector
        # d log_p() / dy

    def log_p(self, x, cond, include_obs=True, method="naive", eps_abs=1e-4, **kwargs):
        """
        The log probability of the observations (X1, y) under the posterior distribution.
        """

        X1 = self.standardize_input_array(cond, **kwargs)
        y = x if isinstance(x, collections.Iterable) else (x,)

        y = np.array(y)
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        if method == "naive":
            K = self.covariance(X1, include_obs=include_obs)
        elif method == "sparse":
            K = self.covariance_spkernel(X1, include_obs=include_obs)
        elif method == "tree":
            K = self.covariance_double_tree(X1, include_obs=include_obs, eps_abs=eps_abs)
        else:
            raise Exception("unknown covariance method %s" % method)
        y = y-self.predict(X1)

        if n==1:
            var = K[0,0]
            ll1 = - .5 * ((y)**2 / var + np.log(2*np.pi*var) )
            return ll1

        L = scipy.linalg.cholesky(K, lower=True)
        ld2 = np.log(np.diag(L)).sum() # this computes .5 * log(det(K))
        alpha = scipy.linalg.cho_solve((L, True), y)
        ll =  -.5 * ( np.dot(y.T, alpha) + n * np.log(2*np.pi)) - ld2

        return ll



    def pack_npz(self):
        d = dict()
        if self.n_features > 0:
            d['beta_bar'] = self.beta_bar
            d['invc'] = self.invc
            d['HKinv'] = self.HKinv
        if self.basis is not None:
            d['basis'] = self.basis
            d.update(self.featurizer_recovery)
        d['X']  = self.X,
        d['y'] =self.y,
        if self.y_obs_variances is not None:
            d['y_obs_variances'] =self.y_obs_variances,
        d['ymean'] = self.ymean,
        d['alpha_r'] =self.alpha_r
        d['Kinv'] =self.Kinv,
        #d['K'] =self.K,
        d['sparse_threshold'] =self.sparse_threshold,
        d['noise_var'] =self.noise_var,
        d['ll'] =self.ll,
        d['n_features'] =self.n_features,

        if self.cov_main is not None:
            d.update(self.cov_main.packed("main"))
        if self.cov_fic is not None:
            d.update(self.cov_fic.packed("fic"))
            d['Luu'] =self.Luu,

        return d

    def __getstate__(self):
        return self.pack_npz()

    def __setstate__(self, d):
        self.unpack_npz(d)
        sparse_invert = scipy.sparse.issparse(self.Kinv)
        self.predict_tree, self.predict_tree_fic = self.build_initial_single_trees(build_single_trees=sparse_invert)
        self.double_tree = None
        self.n = self.X.shape[0]



    def save_trained_model(self, filename):
        """
        Serialize the model to a file.
        """
        d = self.pack_npz()
        with open(filename, 'wb') as f:
            np.savez(f, **d)

    def unpack_npz(self, npzfile):
        self.X = npzfile['X'][0]
        self.y = npzfile['y'][0]
        if 'y_obs_variances' in npzfile:
            self.y_obs_variances = npzfile['y_obs_variances'][0]
        else:
            self.y_obs_variances = None
        self.alpha_r = npzfile['alpha_r'].flatten()
        if 'ymean' in npzfile:
            self.ymean = npzfile['ymean']
        else:
            self.ymean = 0.0

        try:
            # npz mode
            self.noise_var = npzfile['noise_var'].item()
            self.n_features = int(npzfile['n_features'])
        except:
            # dict mode
            self.noise_var = npzfile['noise_var'][0]
            self.n_features = npzfile['n_features'][0]

        self.cov_main = unpack_gpcov(npzfile, 'main')
        self.cov_fic = unpack_gpcov(npzfile, 'fic')
        if self.cov_fic is not None:
            self.Luu = npzfile['Luu'][0]

        self.Kinv = npzfile['Kinv'][0]
        #self.K = npzfile['K'][0]
        self.sparse_threshold = npzfile['sparse_threshold'][0]
        self.ll = npzfile['ll'][0]




        if 'basis' in npzfile:
            self.basis = str(npzfile['basis'])
            self.featurizer, self.featurizer_recovery = recover_featurizer(self.basis, npzfile, transpose=True)
        else:
            self.basis = None
            self.featurizer = None
            self.featurizer_recovery = None

        if self.n_features > 0:
            self.beta_bar = npzfile['beta_bar']
            self.invc = npzfile['invc']
            self.HKinv = npzfile['HKinv']
        else:
            self.HKinv = None

    def load_trained_model(self, filename, build_tree=True, cache_dense=False, leaf_bin_width=0, build_dense_Kinv_hack=False, compile_tree=None):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        del npzfile.f
        npzfile.close()

        self.n = self.X.shape[0]
        sparse_invert = scipy.sparse.issparse(self.Kinv)
        self.predict_tree, self.predict_tree_fic = self.build_initial_single_trees(build_single_trees=sparse_invert)

        if build_tree:
            #self.factor = scikits.sparse.cholmod.cholesky(self.K)
            self.build_point_tree(HKinv = self.HKinv, Kinv=self.Kinv, alpha_r = self.alpha_r, leaf_bin_width=leaf_bin_width, build_dense_Kinv_hack=build_dense_Kinv_hack, compile_tree=compile_tree)
        else:
            self.double_tree = None
        if cache_dense and self.n > 0:
            self.Kinv_dense = self.Kinv.todense()




    def _compute_marginal_likelihood(self, L, z, Binv, H, K, Kinv):

        if scipy.sparse.issparse(L):
            ldiag = L.diagonal()
        else:
            ldiag = np.diag(L)

        z = np.reshape(z, (-1, 1))

        # everything is much simpler in the pure nonparametric case
        if self.n_features == 0:
            ld2_K = np.log(ldiag).sum()
            if np.isnan(ld2_K):
                import pdb; pdb.set_trace()
            self.ll =  -.5 * (np.dot(self.y.T, self.alpha_r) + self.n * np.log(2*np.pi)) - ld2_K

            return

        # keeping explicit CSFIC likelihood code commented out, for debugging.
        # in practice we use the optimized version below
        """
        else:
            Qnn=np.dot(self.K_fic_un.T, np.dot(np.linalg.inv(self.K_fic_uu), self.K_fic_un))
            K_HBH = self.K + Qnn

            K_HBH_inv = np.linalg.inv(K_HBH)

            t1 = np.dot(self.y.T, np.dot(K_HBH_inv, np.reshape(self.y, (-1,1))))
            ld = np.log(np.linalg.det(K_HBH))

            ll1 = -.5 * (t1 + ld + self.n * np.log(2*np.pi))
            self.ll = ll1
            return
        """

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

        tmp1 = Kinv * z
        term1 = np.dot(z.T, tmp1)

        tmp2 = np.dot(self.HKinv, z)
        tmp3 = np.dot(self.invc, tmp2)
        term2 = np.dot(tmp3.T, tmp3)




        # following eqn 2.43 in R&W, we want to compute
        # log det(K + H.T * B * H). using the matrix inversion
        # lemma, we instead compute
        # log det(K) + log det(B) + log det(B^-1 + H*K^-1*H.T)

        # to compute log(det(K)), we use the trick that the
        # determinant of a symmetric pos. def. matrix is the
        # product of squares of the diagonal elements of the
        # Cholesky factor


        ld2_K = np.log(ldiag).sum()
        ld2 =  np.log(np.diag(self.c)).sum() # det( B^-1 - H * K^-1 * H.T )
        ld_B = -np.log(np.linalg.det(Binv))

        # eqn 2.43 in R&W, using the matrix inv lemma
        self.ll = -.5 * (term1 - term2 + self.n * np.log(2*np.pi) + ld_B) - ld2_K - ld2




    # commented-out methods for debugging hyperparam covariances
    """
    def get_dKdi_empirical(self, i, eps=1e-8):
        K = self.predict_tree.kernel_matrix(self.X, self.X, False)

        self.cov_main.dfn_params[i] += eps
        pt = VectorTree(self.X, 1, *self.cov_main.tree_params())
        self.cov_main.dfn_params[i] -= eps
        K1 = pt.kernel_matrix(self.X, self.X, False)

        return (K1 - K) / eps

    def get_dDdi_empirical(self, i, eps=1e-8):
        K = self.predict_tree.kernel_matrix(self.X, self.X, True)

        self.cov_main.dfn_params[i] += eps
        pt = VectorTree(self.X, 1, *self.cov_main.tree_params())
        self.cov_main.dfn_params[i] -= eps
        K1 = pt.kernel_matrix(self.X, self.X, True)

        return (K1 - K) / eps

    def get_dKdi_empirical_fic_wfnvar(self, eps=1e-8):
        self.cov_fic.wfn_params[0] -= eps
        qnn1 = self.build_qnn(self.cov_fic)
        self.cov_fic.wfn_params[0] += 2*eps
        qnn2 = self.build_qnn(self.cov_fic)
        self.cov_fic.wfn_params[0] -= eps
        return (qnn2-qnn1) / (2*eps)

    def get_dKdi_empirical_fic_dfn(self, i, eps=1e-8):
        self.cov_fic.dfn_params[i] -= eps
        qnn1 = self.build_qnn(self.cov_fic)
        self.cov_fic.dfn_params[i] += 2*eps
        qnn2 = self.build_qnn(self.cov_fic)
        self.cov_fic.dfn_params[i] -= eps
        return (qnn2-qnn1) / (2*eps)

    def get_dKdi_empirical_fic_xu(self, p, i, eps=1e-8):
        self.cov_fic.Xu[p,i] -= eps
        qnn1 = self.build_qnn(self.cov_fic)
        self.cov_fic.Xu[p,i] += 2*eps
        qnn2 = self.build_qnn(self.cov_fic)
        self.cov_fic.Xu[p,i] -= eps
        return (qnn2-qnn1) / (2*eps)

    def deriv_uu_wrt_Xu_empirical(p,i, eps=1e-8):
        Xu = np.copy(self.cov_fic.Xu)
        Xu[p,i] -= eps
        K1 = self.predict_tree_fic.kernel_matrix(Xu, Xu, False)
        Xu[p,i] += 2*eps
        K2 = self.predict_tree_fic.kernel_matrix(Xu, Xu, False)
        return (K2-K1)/(2*eps)

    def deriv_un_wrt_Xu_empirical(p,i, eps=1e-8):
        Xu = np.copy(self.cov_fic.Xu)
        Xu[p,i] -= eps
        K1 = self.predict_tree_fic.kernel_matrix(Xu, self.X, False)
        Xu[p,i] += 2*eps
        K2 = self.predict_tree_fic.kernel_matrix(Xu, self.X, False)
        return (K2-K1)/(2*eps)

    """

    def get_dKdi_dense(self, i, n_main_params, n_fic_non_inducing):
        if (i == 0):
            dKdi = np.eye(self.n)
        elif (i == 1):
            if (len(self.cov_main.wfn_params) != 1):
                raise ValueError('gradient computation currently assumes just a single scaling parameter for weight function, but currently wfn_params=%s' % self.cov_main.wfn_params)
            dKdi = self.kernel(self.X, self.X, identical=False) / self.cov_main.wfn_params[0]
        elif i <= n_main_params:
            dKdi = self.predict_tree.kernel_deriv_wrt_i(self.X, self.X, i-2, 1, self.distance_cache_XX)
        elif i == n_main_params+1:
            if (len(self.cov_fic.wfn_params) != 1):
                raise ValueError('gradient computation currently assumes just a single scaling parameter for weight function, but currently wfn_params=%s' % self.cov_fic.wfn_params)
            dKdi = self.get_dKdi_dense_fic_wfnvar()
            #dKdi_empirical = self.get_dKdi_empirical_fic_wfnvar()
        elif i <= n_main_params + n_fic_non_inducing:
            idx = i-n_main_params-2
            dKuu_di = self.predict_tree_fic.kernel_deriv_wrt_i(self.cov_fic.Xu, self.cov_fic.Xu, idx, 1, self.distance_cache_XuXu)
            dKnu_di = self.predict_tree_fic.kernel_deriv_wrt_i(self.X, self.cov_fic.Xu, idx, 0, self.distance_cache_XXu)

            dKdi =  self.get_dKdi_dense_fic(dKuu_di, dKnu_di)

            #dKdi_empirical = self.get_dKdi_empirical_fic_dfn(i-n_main_params-2)

        else:
            assert(target_X is None)
            p = int(np.floor((i-n_main_params-n_fic_non_inducing-1) / self.cov_fic.Xu.shape[1]))
            ii = (i-n_main_params-n_fic_non_inducing-1) % self.cov_fic.Xu.shape[1]
            dKuu_di = self.deriv_uu_wrt_Xu(p,ii)
            dKnu_di = self.deriv_un_wrt_Xu(p,ii).T
            dKdi = self.get_dKdi_dense_fic(dKuu_di, dKnu_di)


            #dKdi_empirical = self.get_dKdi_empirical_fic_xu(p,ii)



        return dKdi

    def get_dKdi_dense_fic_wfnvar(self):
        B = scipy.linalg.solve(self.Luu, self.K_fic_un)
        Qnn = np.dot(B.T, B)
        Qnn /= self.cov_fic.wfn_params[0]
        Qnn += np.diag(1.0 - np.diag(Qnn))
        return Qnn

    def build_qnn(self, cov):
        pt = VectorTree(self.X, 1, *cov.tree_params())
        Kuu = pt.kernel_matrix(cov.Xu, cov.Xu, False)
        Kun = pt.kernel_matrix(cov.Xu, self.X, False)
        Kuu_inv = np.linalg.inv(Kuu)
        Qnn = np.dot(Kun.T, np.dot(Kuu_inv, Kun))
        Qnn += np.diag(cov.wfn_params[0] - np.diag(Qnn))
        return Qnn

    def get_dKdi_dense_fic(self, dKuu_di, dKnu_di):
        tmp = np.dot(dKnu_di, self.D)
        dKdi = tmp + tmp.T - np.dot(self.D.T, np.dot(dKuu_di, self.D))
        dKdi -= np.diag(np.diag(dKdi))
        return dKdi

    def deriv_uu_wrt_Xu(self, p, i):
        nu = self.cov_fic.Xu.shape[0]
        dKdi = np.zeros((nu, nu))

        xp =  np.array(self.cov_fic.Xu[p:p+1,:], copy=True)
        kp = self.predict_tree_fic.kernel_deriv_wrt_xi(xp, self.cov_fic.Xu, 0, i)

        dKdi[p,:] = kp
        dKdi[:,p] = kp
        dKdi[p,p] = 0
        return dKdi

    def deriv_un_wrt_Xu(self, p, i):
        return self.predict_tree_fic.kernel_deriv_wrt_xi(self.cov_fic.Xu, self.X, p, i)


    def get_dlldi_sparse_fic(self, dKuu_di, dKnu_di, alpha, nzr, nzc, Kinv_entries, tmp):
        D = self.D
        # term1 = y^T Kinv (d/dtheta QfuQuu^-1Quf) Kinv y
        nu = self.nu
        term1 = 2 * np.dot(alpha.T, np.dot(dKnu_di, nu)) - \
                    np.dot(nu.T, np.dot(dKuu_di, nu))
        """
        for debugging: we should have
        tmp = np.dot(dKnu_di, D)
        dQ = np.dot(tmp + tmp.T - np.dot(D.T, np.dot(dKuu_di, D)))
        term1 = np.dot(alpha.T, np.dot(dQ, alpha))
        """

        # term2 = y^T Kinv (d/dtheta Lambda) Kinv y
        dD = np.dot(dKuu_di, D)
        dLambda_diag = 2 * np.sum(np.multiply(dKnu_di.T, D), axis=0)
        dLambda_diag -= np.sum(D * dD, axis=0)
        dLambda_diag  = np.reshape(dLambda_diag, alpha.shape)
        term2 = np.dot(alpha.T, np.multiply(dLambda_diag, alpha))

        prod = term1 - term2
        """
        for debugging: should have
        tmp2 = np.dot(dKnu_di, D)
        dKdi = tmp2 + tmp2.T - np.dot(D.T, np.dot(dKuu_di, D))
        dKdi -= np.diag(np.diag(dKdi))
        prod = np.dot(alpha.T, np.dot(dKdi, alpha))
        """

        # now we compute the trace part
        # tr = tr(M * dKdi)
        #     = tr( (Kinv - tmp^Ttmp) (dQ + dLambda)  )
        #     = tr( Kinv (dQ + dLambda) ) - tr(tmp^Ttmp dQ) - tr(tmp^Ttmp dLambda)
        #     = tr1                      - tr2             - tr3
        # where M = (K_cs + Lambda + K_fu K_uu^-1 K_uf)^-1
        #       dQ = d/dtheta  K_fu K_uu^-1 K_uf

        # tr1 = tr( Kinv dQ )
        # compute sum(Kinv elementwise-* dQ) using the sparsity of Kinv
        code = """
        int n = nzc.size();
        int nu = D.shape()(0);
        double sum = 0;
        for (int i=0; i < n; ++i) {
            double entry = 0;
            int ri = nzr(i);
            int ci = nzc(i);

            if(ri == ci) {
               continue;
            }

            for (int k=0; k < nu; ++k) {
                entry -= D(k, ri) * dD(k, ci);
                entry += dKnu_di(ri, k) * D(k, ci);
                entry += dKnu_di(ci, k) * D(k, ri);
            }
            entry *= Kinv_entries(i);
            sum += entry;
        }
        return_val = sum;
        """
        tr1 = weave.inline(code, ['nzc', 'nzr', 'Kinv_entries', 'D', 'dD', 'dKnu_di'],
                           type_converters=converters.blitz, compiler='gcc')

        # tr2 = tr( tmp^T tmp dQ)
        TDt = self.TDt
        T_dnut = np.dot(tmp, dKnu_di)
        TDtduu = np.dot(TDt, dKuu_di)
        tr2 = 2*np.sum(np.multiply(T_dnut, TDt)) - np.sum(np.multiply(TDt, TDtduu))

        # tr3 = tr(tmp^T tmp dLambda)
        diag_tmpTtmp = np.sum(tmp**2, axis=0)
        tr3 = -np.dot(diag_tmpTtmp, dLambda_diag) # negated because our dLambda_diag is negated

        tr = tr1 - tr2 - tr3
        """
        for debugging, we should have:
        tmp2 = np.dot(dKnu_di, D)
        dQdLambda = tmp2 + tmp2.T - np.dot(D.T, np.dot(dKuu_di, D))
        dLambda = np.diag(np.diag(dQdLambda))
        dQ = dQdLambda - dLambda
        tr = np.trace(np.dot(self.Kinv.todense() - np.dot(tmp.T, tmp), dQ))
        """
        dlldi = .5 * (prod-tr)
        return dlldi

    def get_dlldi_sparse(self, i, n_main_params, n_fic_non_inducing, alpha, nzr, nzc, Kinv_entries, tmp):

        if (i == 0):
            # dKdi = scipy.sparse.eye(self.n)


            dlldi = .5 * np.dot(alpha.T, alpha)
            dlldi -= .5 * np.sum(self.Kinv.diagonal())

            if tmp is not None:
                dlldi += .5 * np.sum(tmp**2)

        elif (i == 1):
            if (len(self.cov_main.wfn_params) != 1):
                raise ValueError('gradient computation currently assumes just a single scaling parameter for weight function, but currently wfn_params=%s' % self.cov_main.wfn_params)
            dKdi = self.sparse_kernel(self.X, identical=False) / self.cov_main.wfn_params[0]

            # in general, given dKdi, we want
            #  .5 * alpha^T (  dKdi ) alpha
            # -.5 *  sum(multiply(Kinv - tmp^Ttmp, dKdi )    )
            # where the first line equals
            #  alpha^T Kinv alpha - alpha^T tmp^Ttmp dKdi alpha
            # and the second line equals
            # sum(multiply(Kinv, dKdi)) - sum(multiply( tmp^T tmp, dKdi))

            v1 = dKdi.dot(alpha)
            first_line = .5 * np.dot(alpha.T, v1)

            v6 = self.Kinv.multiply(dKdi).sum()

            if tmp is not None:
                V = dKdi.dot(tmp.T)
                v7 = np.multiply(V, tmp.T).sum()
            else:
                v7 = 0
            second_line = .5 * (v6 - v7)

            dlldi = first_line - second_line


        elif i <= n_main_params:

            K_nzr = np.array(self.distance_cache_XX[:,0], dtype=np.int32, copy=True)
            K_nzc = np.array(self.distance_cache_XX[:,1], dtype=np.int32, copy=True)
            distance_cache = np.array(self.distance_cache_XX[:,2], copy=True)

            entries = self.predict_tree.sparse_kernel_deriv_wrt_i(self.X, self.X, K_nzr, K_nzc, i-2, distance_cache)
            dKdi = scipy.sparse.coo_matrix((entries, (K_nzr, K_nzc)), shape=(self.n, self.n), dtype=float)

            v1 = dKdi.dot(alpha)
            first_line = .5 * np.dot(alpha.T, v1)

            v6 = self.Kinv.multiply(dKdi).sum()

            if tmp is not None:
                V = dKdi.dot(tmp.T)
                v7 = np.multiply(V, tmp.T).sum()
            else:
                v7 = 0
            second_line = .5 * (v6 - v7)

            dlldi = first_line - second_line

        elif i == n_main_params+1:
            if (len(self.cov_fic.wfn_params) != 1):
                raise ValueError('gradient computation currently assumes just a single scaling parameter for weight function, but currently wfn_params=%s' % self.cov_fic.wfn_params)

            B = scipy.linalg.solve(self.Luu, self.K_fic_un)

            # dKdi = B^T B / wfn_params[0] but with diag set to 1.0
            Qnn_diag= self.cov_fic.wfn_params[0] - np.sum(B **2, axis=0)

            Balpha = np.dot(B, alpha)

            first_line = np.dot(Balpha.T, Balpha ) + np.dot(alpha.T, np.multiply(Qnn_diag, np.reshape(np.asarray(alpha), (-1,))))
            first_line *= .5/self.cov_fic.wfn_params[0]

            tr1 = np.sum(np.multiply(self.Kinv.dot(B.T), B.T))
            tr2 = np.dot(self.Kinv.diagonal(), Qnn_diag)


            # now we need
            # tr(tmp^T tmp  (BTB + Qnn_diag))
            # B is m x n
            v = np.dot(tmp, B.T)
            tr3 = np.sum(v**2)
            tr4 = np.dot(np.sum(tmp **2, axis=0), Qnn_diag)


            second_line = .5 * (tr1+tr2 - tr3 - tr4)
            second_line /= self.cov_fic.wfn_params[0]

            dlldi = first_line - second_line

        elif i <= n_main_params + n_fic_non_inducing:
            idx = i-n_main_params-2
            dKuu_di = self.predict_tree_fic.kernel_deriv_wrt_i(self.cov_fic.Xu, self.cov_fic.Xu, idx, 1, self.distance_cache_XuXu)
            dKnu_di = self.predict_tree_fic.kernel_deriv_wrt_i(self.X, self.cov_fic.Xu, idx, 0, self.distance_cache_XXu)
            dlldi =  self.get_dlldi_sparse_fic(dKuu_di, dKnu_di, alpha, nzr, nzc, Kinv_entries, tmp)

        else:
            p = int(np.floor((i-n_main_params-n_fic_non_inducing-1) / self.cov_fic.Xu.shape[1]))
            ii = (i-n_main_params-n_fic_non_inducing-1) % self.cov_fic.Xu.shape[1]
            dKuu_di = self.deriv_uu_wrt_Xu(p,ii)
            dKnu_di = self.deriv_un_wrt_Xu(p,ii).T

            dlldi =  self.get_dlldi_sparse_fic(dKuu_di, dKnu_di, alpha, nzr, nzc, Kinv_entries, tmp)


        return dlldi


    def _log_likelihood_gradient(self, z, Kinv, include_xu=True):
        """
        Gradient of the training set log likelihood with respect to the
        noise variance and the main kernel hyperparams.

        """

        n_main_params = len(self.cov_main.flatten())
        n_fic_non_inducing = 0
        nparams = 1 + n_main_params


        if self.cov_fic is not None:
            nparams += len(self.cov_fic.flatten(include_xu=include_xu))
            n_fic_non_inducing = len(self.cov_fic.flatten(include_xu=False))

        grad = np.zeros((nparams,))

        if not scipy.sparse.issparse(Kinv):
            self.distance_cache_XX = self.predict_tree.kernel_matrix(self.X, self.X, True)
        else:
            max_distance = 1.0 if self.cov_main.wfn_str.startswith("compact") else 1e300
            self.distance_cache_XX = self.predict_tree.sparse_training_kernel_matrix(self.X, max_distance, True)

        if self.n_features > 0:

            tmp = np.dot(self.invc, self.HKinv)
            if scipy.sparse.issparse(Kinv):
                alpha = Kinv.dot(z) - np.dot(tmp.T, np.dot(tmp, z))
            else:
                K_HBH_inv = Kinv - np.dot(tmp.T, tmp)
                alpha = np.matrix(np.reshape(np.dot(K_HBH_inv, z), (-1, 1)))
                M = np.matrix(K_HBH_inv)
            #Qnn=np.dot(self.K_fic_un.T, np.dot(np.linalg.inv(self.K_fic_uu), self.K_fic_un))
            #K_HBH = self.K + Qnn
            #K_HBH_inv = np.linalg.inv(K_HBH)

        else:
            M = Kinv
            alpha = np.matrix(np.reshape(self.alpha_r, (-1,1)))
            tmp = None

        if self.cov_fic is not None:

            self.D = np.asarray(scipy.linalg.solve(self.K_fic_uu, self.K_fic_un))
            self.distance_cache_XuXu = self.predict_tree_fic.kernel_matrix(self.cov_fic.Xu, self.cov_fic.Xu, True)
            self.distance_cache_XXu = self.predict_tree_fic.kernel_matrix(self.X, self.cov_fic.Xu, True)

            self.nu = np.dot(self.D, alpha)
            self.TDt = np.dot(tmp, self.D.T)


        for i in range(nparams):

            if scipy.sparse.issparse(Kinv):
                Kinv_coo = scipy.sparse.coo_matrix(Kinv)
                nzr, nzc, Kinv_entries = Kinv_coo.row, Kinv_coo.col, Kinv_coo.data

                dlldi = self.get_dlldi_sparse(i, n_main_params, n_fic_non_inducing, alpha, nzr, nzc, Kinv_entries, tmp)

            else:
                dKdi = self.get_dKdi_dense(i, n_main_params, n_fic_non_inducing)
                dlldi = .5 * np.dot(alpha.T, np.dot(dKdi, alpha))

                # here we use the fact:
                # trace(AB) = sum_{ij} A_ij * B_ij
                dlldi -= .5 * np.sum(np.sum(np.multiply(M.T, dKdi)))

            grad[i] = dlldi
        return grad


    def log_likelihood(self):
        return self.ll


def treegp_nll_ngrad(**kwargs):
    ll, grad = treegp_ll_grad(**kwargs)
    return -ll, (-grad if grad is not None else np.zeros((len(kwargs['hyperparams']),)))

def treegp_ll_grad(priors=None, **kwargs):

    """
    Get both the log-likelihood and its gradient
    simultaneously (more efficient than doing it separately since we
    only create one new GP object, which only constructs the kernel
    matrix once, etc.).
    """


def optimize_gp_hyperparams(optimize_Xu=True,
                            noise_var=1.0, noise_prior=None,
                            cov_main=None, cov_fic=None,
                            allow_diag=False, **kwargs):

    n_mean_wfn = len(cov_main.wfn_params) if cov_main is not None else 0
    n_mean_dfn = len(cov_main.dfn_params) if cov_main is not None else 0

    n_fic_wfn  = len(cov_fic.wfn_params) if cov_fic is not None else 0
    n_fic_dfn = len(cov_fic.dfn_params) if cov_fic is not None else 0

    n_non_xu = 1 + n_mean_wfn + n_mean_dfn + n_fic_wfn + n_fic_dfn

    bounds = [(1e-8, None),]
    if cov_main is not None:
        bounds += cov_main.bounds()
    if cov_fic is not None:
        bounds += cov_fic.bounds(include_xu=optimize_Xu)

    def covs_from_vector(v):
        i = 0
        noise_var = v[i]
        i += 1
        cm_wfn_params = v[i:i + n_mean_wfn]
        i += n_mean_wfn
        cm_dfn_params = v[i:i + n_mean_dfn]
        i += n_mean_dfn
        cf_wfn_params = v[i:i + n_fic_wfn]
        i += n_fic_wfn
        cf_dfn_params = v[i:i + n_fic_dfn]
        i += n_fic_dfn

        if cov_main is not None:
            new_cov_main = GPCov(wfn_str=cov_main.wfn_str, dfn_str=cov_main.dfn_str,
                                 wfn_params = cm_wfn_params, dfn_params = cm_dfn_params,
                                 wfn_priors = cov_main.wfn_priors, dfn_priors=cov_main.dfn_priors)
        else:
            new_cov_main = None

        if cov_fic is not None:
            if optimize_Xu:
                flat_xu = v[i:]
                Xu = np.reshape(flat_xu, cov_fic.Xu.shape)
            else:
                Xu = cov_fic.Xu

            new_cov_fic = GPCov(wfn_str=cov_fic.wfn_str, dfn_str=cov_fic.dfn_str,
                                 wfn_params = cf_wfn_params, dfn_params = cf_dfn_params,
                                wfn_priors = cov_fic.wfn_priors, dfn_priors=cov_fic.dfn_priors,
                                Xu = Xu)
        else:
            new_cov_fic = None

        return noise_var, new_cov_main, new_cov_fic

    def _nll(v):
        noise_var, new_cov_main, new_cov_fic = covs_from_vector(v)
        gp = GP(compute_ll=True, compute_grad=True, compute_xu_grad =optimize_Xu, noise_var=noise_var,
                cov_main=new_cov_main, cov_fic=new_cov_fic, **kwargs)
        return gp.ll

    def approx_gradient(f, x0, eps):
        n = len(x0)
        grad = np.zeros((n,))
        fx0 = f(x0)
        for i in range(n):
            x_new = x0.copy()
            x_new[i] += eps
            grad[i] = (f(x_new) - fx0) / eps
        return grad

    def nllgrad(v):

        if np.any(v[:n_non_xu] < 1e-10) or not np.all(np.isfinite(v)):
            return np.float('inf'), np.zeros(v.shape)

        noise_var, new_cov_main, new_cov_fic = covs_from_vector(v)

        try:
            gp = GP(compute_ll=True, compute_grad=True, compute_xu_grad =optimize_Xu, noise_var=noise_var,
                    cov_main=new_cov_main, cov_fic=new_cov_fic, **kwargs)
            ll = gp.ll
            grad = gp.ll_grad
            del gp

            ll += noise_prior.log_p(noise_var) + \
                  ( new_cov_main.prior_logp() if new_cov_main is not None else 0 ) + \
                  ( new_cov_fic.prior_logp() if new_cov_fic is not None else 0 )
            prior_grad = np.concatenate([[noise_prior.deriv_log_p(noise_var)],
                                    new_cov_main.prior_grad() if new_cov_main is not None else [],
                                    new_cov_fic.prior_grad(include_xu=optimize_Xu) if new_cov_fic is not None else []])
            grad += prior_grad




        except FloatingPointError as e:
            print "warning: floating point error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad = np.zeros((len(v),))
        except np.linalg.linalg.LinAlgError as e:
            print "warning: lin alg error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad = np.zeros((len(v),))
        except scikits.sparse.cholmod.CholmodError as e:
            print "warning: cholmod error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad = np.zeros((len(v),))
        #except ValueError as e:
        #    print "warning: value error (%s) in likelihood computation, returning likelihood -inf" % str(e)
        #    ll = np.float("-inf")
        #    grad = np.zeros((len(v),))
        #print "hyperparams", v, "ll", ll, 'grad', grad

        if np.isnan(grad).any():
            raise Exception('nanana')

        return -1 * ll, (-1 * grad  if grad is not None else None)



    def build_gp(v, **kwargs2):
        noise_var, new_cov_main, new_cov_fic = covs_from_vector(v)
        kw = dict(kwargs.items() + kwargs2.items())
        gp = GP(noise_var=noise_var, cov_main=new_cov_main, cov_fic=new_cov_fic, **kw)
        return gp

    x0 = np.concatenate([[noise_var,],
                         cov_main.flatten() if cov_main is not None else [],
                         cov_fic.flatten(include_xu = optimize_Xu) if cov_fic is not None else []])
    return nllgrad, x0, bounds, build_gp, covs_from_vector
