import os
import time
import numpy as np
import collections
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scikits.sparse.cholmod
import sklearn.preprocessing
import pyublas
import hashlib
import types
import marshal


from sparsegp.cover_tree import VectorTree, MatrixTree

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
    t0 = time.time()
    entries = tree.sparse_training_kernel_matrix(X, max_distance)
    spK = scipy.sparse.coo_matrix((entries[:,2], (entries[:,0], entries[:,1])), shape=(n,n), dtype=float)
    t1 = time.time()
    print "sparse kernel", t1-t0

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

def prior_sample(X, hyperparams, dfn_str, wfn_str, sparse_threshold=1e-20, return_kernel=False):
    n = X.shape[0]
    noise_var, dfn_params, wfn_params = extract_hyperparams(dfn_str, wfn_str, hyperparams)
    predict_tree = VectorTree(X, 1, dfn_str, dfn_params, wfn_str, wfn_params)

    spK = sparse_kernel_from_tree(predict_tree, X, sparse_threshold, True, noise_var)
    factor = scikits.sparse.cholmod.cholesky(spK)
    L = factor.L()
    P = factor.P()
    Pinv = np.argsort(P)
    z = np.random.randn(n)
    y = np.array((L * z)[Pinv]).reshape((-1,))
    if return_kernel:
        return y, spK
    else:
        return y

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

class GPCov(object):
    def __init__(self, wfn_params, dfn_params,
                 wfn_str="se", dfn_str="euclidean",
                 wfn_priors=None, dfn_priors=None,
                 Xu=None):
        self.wfn_str =str(wfn_str)
        self.wfn_params = np.array(wfn_params, dtype=float)
        self.dfn_str = str(dfn_str)
        self.dfn_params = np.array(dfn_params, dtype=float)
        self.Xu = Xu

        self.wfn_priors = wfn_priors
        self.dfn_priors = dfn_priors
        if self.wfn_priors is None:
            self.wfn_priors = [None,] * len(self.wfn_params)
        if self.dfn_priors is None:
            self.dfn_priors = [None,] * len(self.dfn_params)

    def tree_params(self):
        return (self.dfn_str, self.dfn_params, self.wfn_str, self.wfn_params)

    def packed(self, prefix):
        return {prefix + "_wfn_str": self.wfn_str,
                prefix + "_wfn_params": self.wfn_params,
                prefix + "_dfn_str": self.dfn_str,
                prefix + "_dfn_params": self.dfn_params,
                prefix + "_Xu": self.Xu}

    def flatten(self, include_xu=True):
        v = np.concatenate([self.wfn_params, self.dfn_params])
        if include_xu and self.Xu is not None:
            v = np.concatenate([v, self.Xu.flatten()])
        return v

    def prior_logp(self):
        return prior_ll(self.dfn_params, self.dfn_priors) + prior_ll(self.wfn_params, self.wfn_priors)

    def prior_grad(self, include_xu=True):
        v = np.concatenate([prior_grad(self.dfn_params, self.dfn_priors) ,
                            prior_grad(self.wfn_params, self.wfn_priors)])
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

    def build_kernel_matrix(self, X):
        K = self.kernel(X, X, identical=True)
        return K + np.eye(K.shape[0], dtype=np.float64) * 1e-8 # try to avoid losing
                                       # positive-definiteness
                                       # to numeric issues

    def sparse_build_kernel_matrix(self, X):
        K = self.sparse_kernel(X, identical=True)
        K = K + scipy.sparse.eye(K.shape[0], dtype=np.float64) * 1e-8 # try to avoid losing
                                       # positive-definiteness
                                       # to numeric issues
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

        #unpermuted_L = factor.L()
        #P = factor.P()
        #Pinv = np.argsort(P)
        #L = unpermuted_L[Pinv,:]
        alpha = factor(self.y)
        t2 = time.time()
        self.timings['solve_alpha'] = t2-t1
        Kinv = factor(scipy.sparse.eye(K.shape[0]).tocsc())
        t3 = time.time()
        self.timings['solve_Kinv'] = t3-t2

        I = (Kinv.getrow(0) * K.getcol(0)).todense()[0,0]
        if np.abs(I - 1) > 0.01:
            print "WARNING: poorly conditioned inverse (I=%f)" % I

        return alpha, factor, factor.L(), Kinv

    def build_parametric_model(self, alpha, Kinv_sp, H, b, B):
        # notation follows section 2.7 of Rasmussen and Williams
        b = np.reshape(b, (-1,))
        Binv = scipy.linalg.inv(B)
        tmp = np.reshape(np.asarray(np.dot(H, alpha)), (-1,)) + np.dot(Binv, b)  # H * K^-1 * y + B^-1 * b

        HKinv = H * Kinv_sp
        M_inv  = Binv + np.dot(HKinv, H.T) # here M = (inv(B) +
                                           # H*K^-1*H.T)^-1 is the
                                           # posterior covariance
                                           # matrix on the params.

        c = scipy.linalg.cholesky(M_inv, lower=True) # c = sqrt(inv(B) + H*K^-1*H.T)
        beta_bar = scipy.linalg.cho_solve((c, True), tmp)
        invc = scipy.linalg.inv(c)

        return c, beta_bar, invc, HKinv

    def get_data_features(self, X):
        H = np.array([[f(x) for x in X] for f in self.basisfns], dtype=float)
        return H

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
            return scipy.sparse.csc_matrix(np.asarray(M) * (np.abs(M) > self.sparse_threshold))

    def sort_events(self, X, y):
        combined = np.hstack([X, np.reshape(y, (-1, 1))])
        combined_sorted = np.array(sorted(combined, key = lambda x: x[0]), dtype=float)
        X_sorted = np.array(combined_sorted[:, :-1], copy=True, dtype=float)
        y_sorted = combined_sorted[:, -1].flatten()
        return X_sorted, y_sorted

    def __init__(self, X=None, y=None,
                 fname=None,
                 noise_var=1.0,
                 cov_main=None,
                 cov_fic=None,
                 basisfns=(),
                 param_mean=None, param_cov=None,
                 compute_ll=False,
                 compute_grad=False,
                 sparse_threshold=1e-10,
                 K = None,
                 dfn_str = "lld",
                 wfn_str = "se",
                 sort_events=False,
                 build_tree=True,
                 sparse_invert=True,
                 center_mean=False,
                 leaf_bin_size = 0): # WARNING: bin sizes > 0 currently lead to memory leaks


        self.double_tree = None
        if fname is not None:
            self.load_trained_model(fname, build_tree=build_tree, leaf_bin_size=leaf_bin_size)
        else:
            if sort_events:
                X, y = self.sort_events(X, y) # arrange events by
                                              # lon/lat, as a
                                              # heuristic to expose
                                              # block structure in the
                                              # kernel matrix


            self.cov_main = cov_main
            self.cov_fic = cov_fic
            self.noise_var = noise_var
            self.sparse_threshold = sparse_threshold
            self.basisfns = basisfns
            self.timings = dict()

            if X is not None:
                self.X = np.matrix(X, dtype=float)
                self.y = np.array(y, dtype=float)
                if center_mean:
                    self.ymean = np.mean(y)
                    self.y -= self.ymean
                else:
                    self.ymean = 0.0
                self.n = X.shape[0]
            else:
                self.X = np.reshape(np.array(()), (0,0))
                self.y = np.reshape(np.array(()), (0,))
                self.n = 0
                self.ymean = 0.0
                self.K = np.reshape(np.array(()), (0,0))
                self.Kinv = np.reshape(np.array(()), (0,0))
                self.alpha_r = self.y
                self.ll = np.float('-inf')
                return

            H = self.get_data_features(X)

            # train model
            t0 = time.time()
            if build_tree:
                tree_X = pyublas.why_not(self.X)
            else:
                tree_X = np.array([[0.0,],], dtype=float)
            self.predict_tree = VectorTree(tree_X, 1, *self.cov_main.tree_params())
            t1 = time.time()
            self.timings['build_predict_tree'] = t1-t0

            if K is None:
                if sparse_invert and build_tree:
                    K = self.sparse_build_kernel_matrix(self.X)
                else:
                    K = self.build_kernel_matrix(self.X)
            self.K = self.sparsify(K)

            if sparse_invert:
                alpha, factor, L, Kinv = self.sparse_invert_kernel_matrix(self.K)
                self.factor = factor
            else:
                alpha, factor, L, Kinv = self.invert_kernel_matrix(K)
            self.Kinv = self.sparsify(Kinv)

            if len(self.basisfns) > 0:
                self.c,self.beta_bar, self.invc, self.HKinv = self.build_parametric_model(alpha,
                                                                                          self.Kinv,
                                                                                          H,
                                                                                          b=param_mean,
                                                                                          B=param_cov)
                r = self.y - np.dot(H.T, self.beta_bar)
                z = np.dot(H.T, param_mean) - self.y
                B = param_cov
            else:
                self.HKinv = None
                r = self.y
                z = self.y
                B = None

            self.alpha_r = np.reshape(np.asarray(factor(r)), (-1,))

            if build_tree:
                self.build_point_tree(HKinv = self.HKinv, Kinv=self.Kinv, alpha_r = self.alpha_r, leaf_bin_size=leaf_bin_size)
            #t6 = time.time()

            # precompute training set log likelihood, so we don't need
            # to keep L around.
            if compute_ll:
                self._compute_marginal_likelihood(L=L, z=z, B=B, H=H, K=self.K, Kinv=self.Kinv)
            else:
                self.ll = -np.inf
            #t7 = time.time()
            if compute_grad:
                self.ll_grad = self._log_likelihood_gradient(z=z, H=H, B=B, Kinv=self.Kinv)

            #t8 = time.time()
            """
            print t1-t0
            print t2-t1
            print t3-t2
            print t4-t3
            print t5-t4
            print t6-t5
            print t7-t6
            print t8-t7
            """

    def build_point_tree(self, HKinv, Kinv, alpha_r, leaf_bin_size):
        if self.n == 0: return

        fullness = len(self.Kinv.nonzero()[0]) / float(self.Kinv.shape[0]**2)
        print "Kinv is %.1f%% full." % (fullness * 100)
        #if fullness > .15:
        #    raise Exception("not building tree, Kinv is too full!" )

        self.predict_tree.set_v(0, alpha_r.astype(np.float))

        d = len(self.basisfns)
        if d > 0:
            self.cov_tree = VectorTree(self.X, d, *self.cov_main.tree_params())
            HKinv = HKinv.astype(np.float)
            for i in range(d):
                self.cov_tree.set_v(i, HKinv[i, :])


        nzr, nzc = Kinv.nonzero()
        vals = np.reshape(np.asarray(Kinv[nzr, nzc]), (-1,))
        self.double_tree = MatrixTree(self.X, nzr, nzc, *self.cov_main.tree_params())
        self.double_tree.set_m_sparse(nzr, nzc, vals)
        if leaf_bin_size > 1:
            self.double_tree.collapse_leaf_bins(leaf_bin_size)

    def predict(self, cond, parametric_only=False, eps=1e-8):
        if not self.double_tree: return self.predict_naive(cond, parametric_only)
        X1 = self.standardize_input_array(cond).astype(np.float)

        if parametric_only:
            gp_pred = np.zeros((X1.shape[0],))
        else:
            gp_pred = np.array([self.predict_tree.weighted_sum(0, np.reshape(x, (1,-1)), eps) for x in X1])

        if len(self.basisfns) > 0:
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

        if len(self.basisfns) > 0:
            H = self.get_data_features(X1)
            mean_pred = np.reshape(np.dot(H.T, self.beta_bar), gp_pred.shape)
            gp_pred += mean_pred

        if len(gp_pred) == 1:
            gp_pred = gp_pred[0]

        gp_pred += self.ymean
        return gp_pred


    def kernel(self, X1, X2, identical=False):
        K = self.predict_tree.kernel_matrix(X1, X2, False)
        if identical:
            K += self.noise_var * np.eye(K.shape[0])
        return K

    def sparse_kernel(self, X, identical=False):
        if self.sparse_threshold ==0:
            max_distance = 1e300
        else:
            max_distance = np.sqrt(-np.log(self.sparse_threshold)) # assuming a SE kernel
        entries = self.predict_tree.sparse_training_kernel_matrix(X, max_distance)
        spK = scipy.sparse.coo_matrix((entries[:,2], (entries[:,1], entries[:,0])), shape=(self.n, len(X)), dtype=float)
        if identical:
            spK = spK + self.noise_var * scipy.sparse.eye(spK.shape[0])
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

            if self.basisfns:
                H = self.get_data_features(X1)
                self.querysp_R = H - np.asmatrix(self.HKinv) * self.querysp_K

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.querysp_K

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
                H = self.get_data_features(X1)
                self.query_R = H - np.asmatrix(self.HKinv) * self.query_K

#            print "cache fail: model %d called with " % (len(self.alpha)), X1
#        else:
#            print "cache hit!"
        return self.query_K

    def covariance_spkernel(self, cond, include_obs=False, parametric_only=False, pad=1e-8):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        Kstar = self.get_query_K_sparse(X1)
        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:
                tmp = self.Kinv * Kstar
                qf = (Kstar.T * tmp).todense()
                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))

        if len(self.basisfns) > 0:
            R = self.querysp_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])

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

        if len(self.basisfns) > 0:
            R = self.querysp_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])

        return gp_cov

    def covariance(self, cond, include_obs=False, parametric_only=False, pad=1e-8):
        """
        Compute the posterior covariance matrix at a set of points given by the rows of X1.

        Default is to compute the covariance of f, the latent function values. If obs_covar
        is True, we instead compute the covariance of y, the observed values.

        By default, we add a tiny bit of padding to the diagonal to counteract any potential
        loss of positive definiteness from numerical issues. Setting pad=0 disables this.

        """
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]

        Kstar = self.get_query_K(X1)
        if not parametric_only:
            gp_cov = self.kernel(X1,X1, identical=include_obs)
            if self.n > 0:
                tmp = self.Kinv * Kstar
                qf = np.dot(Kstar.T, tmp)
                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))

        if len(self.basisfns) > 0:
            R = self.query_R
            tmp = np.dot(self.invc, R)
            mean_cov = np.dot(tmp.T, tmp)
            gp_cov += mean_cov

        gp_cov += pad * np.eye(gp_cov.shape[0])

        return gp_cov

    def covariance_double_tree(self, cond, include_obs=False, parametric_only=False, pad=1e-8, eps=1e-8, eps_abs=1e-4, cutoff_rule=2):
        X1 = self.standardize_input_array(cond)
        m = X1.shape[0]
        d = len(self.basisfns)

        if not parametric_only:
            gp_cov = self.kernel(X1, X1, identical=include_obs)
            if self.n > 0:
                qf = self.double_tree.quadratic_form(X1, X1, eps, eps_abs, cutoff_rule=cutoff_rule)
                gp_cov -= qf
        else:
            gp_cov = np.zeros((m,m))

        if len(self.basisfns) > 0:
            H = np.array([[f(x) for x in X1] for f in self.basisfns], dtype=np.float64)
            HKinvKstar = np.zeros((d, m))

            for i in range(d):
                for j in range(m):
                    HKinvKstar[i,j] = self.cov_tree.weighted_sum(i, X1[j:j+1,:], eps)
            R = H - HKinvKstar
            v = np.dot(self.invc, R)
            mc = np.dot(v.T, v)
            gp_cov += mc

        gp_cov += pad * np.eye(m)
        return gp_cov

    def variance(self,cond, **kwargs):
        return np.diag(self.covariance(cond, **kwargs))

    def sample(self, cond, include_obs=True):
        """
        Sample from the GP posterior at a set of points given by the rows of X1.

        Default is to sample observed values (i.e. we include observation noise). If obs=False, we instead
        sample values of the latent function f.
        """

        X1 = self.standardize_input_array(cond)
        (n,d) = X1.shape
        means = np.reshape(self.predict(X1), (-1, 1))
        K = self.covariance(X1, include_obs=include_obs)
        samples = np.random.randn(n, 1)

        L = scipy.linalg.cholesky(K, lower=True)
        samples = means + np.dot(L, samples)


        if len(samples) == 1:
            samples = samples[0]

        return samples

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

    def log_p(self, x, cond, include_obs=True, **kwargs):
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

        K = self.covariance_spkernel(X1, include_obs=include_obs)
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
        if self.basisfns is not None and len(self.basisfns) > 0:
            d['c'] = self.c
            d['beta_bar'] = self.beta_bar
            d['invc'] = self.invc
            d['HKinv'] = self.HKinv
            d['basisfns'] = np.array([marshal_fn(f) for f in self.basisfns], dtype=object)
        else:
            d['basisfns'] = np.empty(0)
        d['X']  = self.X,
        d['y'] =self.y,
        d['ymean'] = self.ymean,
        d['alpha_r'] =self.alpha_r,
        d['Kinv'] =self.Kinv,
        d['K'] =self.K,
        d['sparse_threshold'] =self.sparse_threshold,
        d['noise_var'] =self.noise_var,
        d['ll'] =self.ll,
        d['alpha_r'] = self.alpha_r

        if self.cov_main is not None:
            d.update(self.cov_main.packed("main"))
        if self.cov_fic is not None:
            d.update(self.cov_fic.packed("fic"))

        return d

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
        if 'ymean' in npzfile:
            self.ymean = npzfile['ymean']
        else:
            self.ymean = 0.0

        self.Kinv = npzfile['Kinv'][0]
        self.K = npzfile['K'][0]
        self.sparse_threshold = npzfile['sparse_threshold'][0]
        self.ll = npzfile['ll'][0]

        self.noise_var = npzfile['noise_var'][0]
        self.cov_main = unpack_gpcov(npzfile, 'main')
        self.cov_fic = unpack_gpcov(npzfile, 'fic')

        self.basisfns = npzfile['basisfns']
        if self.basisfns is not None and len(self.basisfns) > 0:
            self.basisfns = [unmarshal_fn(code) for code in self.basisfns]
            self.beta_bar = npzfile['beta_bar']
            self.c = npzfile['c']
            self.invc = npzfile['invc']
            self.HKinv = npzfile['HKinv']
        else:
            self.HKinv = None
        self.alpha_r = npzfile['alpha_r']

    def load_trained_model(self, filename, build_tree=True, cache_dense=False, leaf_bin_size=0):
        npzfile = np.load(filename)
        self.unpack_npz(npzfile)
        del npzfile.f
        npzfile.close()

        self.n = self.X.shape[0]
        self.predict_tree = VectorTree(self.X[0:2,:], 1, *self.cov_main.tree_params())
        if build_tree:
            self.factor = scikits.sparse.cholmod.cholesky(self.K)
            self.build_point_tree(HKinv = self.HKinv, Kinv=self.Kinv, alpha_r = self.alpha_r, leaf_bin_size=leaf_bin_size)
        if cache_dense and self.n > 0:
            self.Kinv_dense = self.Kinv.todense()

    def _compute_marginal_likelihood(self, L, z, B, H, K, Kinv):

        if scipy.sparse.issparse(L):
            ldiag = L.diagonal()
        else:
            ldiag = np.diag(L)

        # everything is much simpler in the pure nonparametric case
        if not self.basisfns:
            ld2_K = np.log(ldiag).sum()
            self.ll =  -.5 * (np.dot(self.y.T, self.alpha_r) + self.n * np.log(2*np.pi)) - ld2_K
            return

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
        ld_B = np.log(np.linalg.det(B))

        # eqn 2.43 in R&W, using the matrix inv lemma
        self.ll = -.5 * (term1 - term2 + self.n * np.log(2*np.pi) + ld_B) - ld2_K - ld2

    def get_dKdi_dense(self, X, i):
        if (i == 0):
            dKdi = np.eye(self.n)
        else:
            dKdi = self.predict_tree.kernel_deriv_wrt_i(self.X, self.X, i-1)
        return dKdi

    def get_dKdi_sparse(self, X, i, M):
        if (i == 0):
            dKdi = scipy.sparse.eye(self.n)
        else:
            nzr, nzc = M.nonzero()
            entries = self.predict_tree.sparse_kernel_deriv_wrt_i(self.X, self.X, nzr, nzc, i-1)
            dKdi = scipy.sparse.coo_matrix((entries, (nzr, nzc)), shape=(self.n, self.n), dtype=float)
        return dKdi

    def _llgrad_main(self, z, H, B, Kinv):
        """
        Gradient of the training set log likelihood with respect to the
        noise variance and the main kernel hyperparams.

        """

        nparams = 1 + len(self.cov_main.wfn_params) + len(self.cov_main.dfn_params)
        grad = np.zeros((nparams,))

        if self.basisfns:
            tmp = np.dot(self.invc, self.HKinv)
            K_HBH_inv = Kinv - np.dot(tmp.T, tmp)
            alpha_z = np.reshape(np.dot(K_HBH_inv, z), (-1, 1))

            M = np.matrix(K_HBH_inv)
            alpha = np.matrix(np.reshape(alpha_z, (-1,1)))
        else:
            M = Kinv
            alpha = np.matrix(np.reshape(self.alpha_r, (-1,1)))

        for i in range(nparams):
            if scipy.sparse.issparse(M):
                tA = time.time()
                dKdi = self.get_dKdi_sparse(self.X, i, M)
                dlldi = .5 * alpha.T * dKdi * alpha
                tB = time.time()

                # here we use the fact:
                # trace(AB) = sum_{ij} A_ij * B_ij
                dlldi -= .5 * M.T.multiply(dKdi).sum()
                tC = time.time()

            else:
                tA = time.time()
                dKdi = self.get_dKdi_dense(self.X, i)
                dlldi = .5 * np.dot(alpha.T, np.dot(dKdi, alpha))
                tB = time.time()

                # here we use the fact:
                # trace(AB) = sum_{ij} A_ij * B_ij
                dlldi -= .5 * np.sum(np.sum(np.multiply(M.T, dKdi)))
                tC = time.time()

            grad[i] = dlldi

        return grad

    def _log_likelihood_gradient(self, z, H, B, Kinv):
        return self._llgrad_main(z, H, B, Kinv)

    def log_likelihood(self):
        return self.ll


def sparsegp_nll_ngrad(**kwargs):
    ll, grad = sparsegp_ll_grad(**kwargs)
    return -ll, (-grad if grad is not None else np.zeros((len(kwargs['hyperparams']),)))

def sparsegp_ll_grad(priors=None, **kwargs):

    """
    Get both the log-likelihood and its gradient
    simultaneously (more efficient than doing it separately since we
    only create one new GP object, which only constructs the kernel
    matrix once, etc.).
    """


def optimize_gp_hyperparams(optimize_Xu=True,
                            noise_var=1.0, noise_prior=None,
                            cov_main=None, cov_fic=None, **kwargs):

    n_mean_wfn = len(cov_main.wfn_params) if cov_main is not None else 0
    n_mean_dfn = len(cov_main.dfn_params) if cov_main is not None else 0

    n_fic_wfn  = len(cov_fic.wfn_params) if cov_fic is not None else 0
    n_fic_dfn = len(cov_fic.dfn_params) if cov_fic is not None else 0

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
                                 wfn_params = cm_wfn_params, dfn_params = cm_dfn_params)
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
                                Xu = Xu)
        else:
            new_cov_fic = None

        return noise_var, new_cov_main, new_cov_fic


    def nllgrad(v):
        noise_var, new_cov_main, new_cov_fic = covs_from_vector(v)

        try:
            gp = GP(compute_ll=True, compute_grad=True, noise_var=noise_var,
                    cov_main=new_cov_main, cov_fic=new_cov_fic, **kwargs)
            ll = gp.ll
            grad = gp.ll_grad

            ll += noise_prior.log_p(noise_var) + \
                  ( new_cov_main.prior_logp() if new_cov_main is not None else 0 ) + \
                  ( new_cov_fic.prior_logp() if new_cov_fic is not None else 0 )
            grad += np.concatenate([[noise_prior.deriv_log_p(noise_var)],
                                    new_cov_main.prior_grad() if new_cov_main is not None else [],
                                    new_cov_fic.prior_grad(include_xu=optimize_Xu) if new_cov_fic is not None else []])

        except FloatingPointError as e:
            print "warning: floating point error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad = np.zeros((len(v),))
            raise
        except np.linalg.linalg.LinAlgError as e:
            print "warning: lin alg error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad = np.zeros((len(v),))
        except scikits.sparse.cholmod.CholmodError as e:
            print "warning: cholmod error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad = np.zeros((len(v),))
        except ValueError as e:
            print "warning: value error (%s) in likelihood computation, returning likelihood -inf" % str(e)
            ll = np.float("-inf")
            grad = np.zeros((len(v),))
        print "hyperparams", v, "ll", ll, "grad", grad
        return -1 * ll, (-1 * grad  if grad is not None else None)

    def build_gp(v, **kwargs2):
        noise_var, new_cov_main, new_cov_fic = covs_from_vector(v)
        kw = dict(kwargs.items() + kwargs2.items())
        gp = GP(noise_var=noise_var, cov_main=new_cov_main, cov_fic=new_cov_fic, **kw)
        return gp

    x0 = np.concatenate([[noise_var,],
                         cov_main.flatten() if cov_main is not None else [],
                         cov_fic.flatten(include_xu = optimize_Xu) if cov_fic is not None else []])
    return nllgrad, x0, build_gp, covs_from_vector
