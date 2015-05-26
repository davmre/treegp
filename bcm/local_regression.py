import numpy as np
import scipy.stats
from treegp.gp import GP, GPCov, mcov, dgaussian, dgaussian_rank1
from collections import defaultdict

from multiprocessing import Pool


class LocalGPs(object):

    def __init__(self, block_centers, cov_block_params, X=None, y=None, X_blocks=None, y_blocks=None, block_idxs=None, cov_blocks=None, gps=None, compute_neighbors=False):
        self.n_blocks = len(block_centers)
        self.block_centers = block_centers

        if X is not None:
            X_blocks, y_blocks, block_idxs = self.blocks_from_centers(X, y)
        self.X_blocks = X_blocks
        self.y_blocks = y_blocks
        self.block_idxs = block_idxs
        self.cov_block_params = cov_block_params
        self.y = y

        if cov_blocks is None:
            cov_blocks = []
            for cbp in cov_block_params:
                noise_var = cbp[0]
                cov = GPCov(wfn_params=cbp[1:2], dfn_params=cbp[2:], dfn_str="euclidean", wfn_str="se")
                cov_blocks.append((noise_var, cov))

        if len(cov_blocks) == 1:
            cov_blocks = cov_blocks * self.n_blocks
            self.tied_params = True
        else:
            self.tied_params = False

        self.cov_blocks = cov_blocks

        if gps is None:
            gps = []
            for (X, y, nc) in zip(X_blocks, y_blocks, cov_blocks):
                noise_var, cov = nc
                if len(y) > 0:
                    gp = GP(X, y, cov_main=cov,
                            noise_var=noise_var,
                            compute_ll=True,
                            compute_grad=True,
                            sort_events=False,
                            sparse_invert=False)
                    gps.append(gp)
                else:
                    gps.append(None)
        self.gps = gps

        if compute_neighbors:
            self.compute_neighbors()

    def blocks_from_centers(self, X, y):
        assert(len(y) == X.shape[0])
        X_blocks = [[] for i in range(self.n_blocks)]
        y_blocks = [[] for i in range(self.n_blocks)]
        idxs = [[] for i in range(self.n_blocks)]
        for i, (xp, yp) in enumerate(zip(X, y)):
            block = self.get_block(xp)
            X_blocks[block].append(xp)
            y_blocks[block].append(yp)
            idxs[block].append(i)
        X_blocks = [np.array(xb) for xb in X_blocks]
        y_blocks = [np.array(yb) for yb in y_blocks]

        return X_blocks, y_blocks, idxs

    def unscramble_Xblocks(self, X_blocks):
        d = self.get_dimension()
        n = np.sum([xb.shape[0] for xb in X_blocks])
        X = np.zeros((n, d))
        for xb, idxs in zip(X_blocks, self.block_idxs):
            for x, idx in zip(xb, idxs):
                X[idx, :] = x
        return X

    def get_block(self, X_new):
        dists = [np.linalg.norm(X_new - center) for center in self.block_centers]
        return np.argmin(dists)

    def likelihood(self):
        return np.sum([gp.log_likelihood() for gp in self.gps if gp is not None])

    def llgrad_hparam(self):
        ll = self.likelihood()
        zerograd = np.zeros(self.cov_block_params[0].shape)
        grads = [gp.ll_grad if gp is not None else zerograd for gp in self.gps ]
        if self.tied_params:
            return ll, np.sum(grads, axis=0)
        else:
            return ll, np.concatenate(grads)

    def llgrad_X(self):
        block_grads = [gp.grad_ll_wrt_X() if gp is not None else np.array([]) for gp in self.gps]
        grads = self.unscramble_Xblocks(block_grads)
        return grads

    def predict(self, X_new):
        p = []
        for xn in X_new:
            block = self.get_block(xn)
            p.append(self.gps[block].predict(xn.reshape(1, -1)))
        return np.array(p)

    def covariance(self, X_new):

        Xs, perm, b = self.sort_by_block(X_new)

        nx = len(perm)
        cov = np.zeros((nx, nx))

        i_old = 0
        for block, i in enumerate(b):
            Xs_block = Xs[i_old:i,:]
            cov[i_old:i, i_old:i] = self.gps[block].covariance(Xs_block)
            i_old = i

        rperm = np.argsort(perm)
        tmp = cov[:, rperm]
        rcov = tmp[rperm, :]

        return rcov

    def predict_dist(self, X_new):
        return self.predict(X_new), self.covariance(X_new)

    def sort_by_block(self, X):
        # given: set of input locations X
        # returns: tuple (Xs, p, b)
        # Xs: X locations sorted by block, i.e. first all points in block 0,
        #     then block 1, etc.
        # p: permutation such that Xs = X[p, :]
        # b: endpoints of each block in Xs. For example, if we have three
        #    points in block 0, none in block 1, and one in block 2, we'd
        #    return b = [3, 3, 4]
        blocks = np.array([self.get_block(xn) for xn in X])
        idx = np.arange(len(blocks))
        perm = np.array(sorted(idx, key = lambda i : blocks[i]))

        sorted_blocks = blocks[perm]
        b = []
        i = 0
        for block in range(self.n_blocks):
            i += np.sum(sorted_blocks==block)
            b.append(i)

        return X[perm,:], perm, b


    def flat_X(self):
        return np.concatenate([x.flatten() for x in self.X_blocks])


    def get_dimension(self):
        d = None
        i = 0
        while d is None and i < len(self.X_blocks):
            try:
                d = self.X_blocks[i].shape[1]
            except:
                i += 1
                continue
        return d

    def update_X(self, flatX):
        # assign each new X location to the block for the corresponding region
        # (even if this is different than the block this X was previously in)

        newX = flatX.reshape((-1, self.get_dimension() ))

        return LocalGPs(block_centers=self.block_centers,
                        cov_block_params = self.cov_block_params,
                        X = newX,
                        y = self.y)


class BCM(LocalGPs):

    def __init__(self, test_cov, **kwargs):
        super(BCM, self).__init__(**kwargs)
        self.test_cov = test_cov

    def predict_dist(self, X_new, diag=False, source_gps=None, predict_cov=None, noise_var=0.0):

        if diag:
            means = []
            vs = []
            for x in X_new:
                m, v = self.predict_dist(x.reshape((1, -1)), diag=False, source_gps=source_gps, noise_var=noise_var)
                means.append(m)
                vs.append(v)
            return np.array(means), np.array(vs)

        if source_gps is None:
            source_gps = self.gps

        if predict_cov is None:
            predict_cov = self.test_cov


        means = [lgp.predict(X_new) for lgp in source_gps]
        covs = [lgp.covariance(X_new, include_obs=True) for lgp in source_gps]
        precs = [np.linalg.inv(cov) for cov in covs]
        combined_prec = np.sum(precs, axis=0)

        target_prior_cov = mcov(X_new, predict_cov, noise_var)
        target_prior_prec = np.linalg.inv(target_prior_cov)
        prior_covs = [mcov(X_new, lgp.cov_main, lgp.noise_var) for lgp in source_gps]
        prior_precs = [np.linalg.inv(cov) for cov in prior_covs]
        combined_prec += target_prior_prec
        combined_prec -= np.sum(prior_precs, axis=0)
        cov = np.linalg.inv(combined_prec)

        assert( (np.diag(cov) > 0).all() )

        weighted_means = [np.dot(prec, mean) for (mean, prec) in zip(means, precs)]
        mean = np.dot(cov, np.sum(weighted_means, axis=0))
        return mean, cov

    def _local_pred_gradient_Xi_target(self, target_i, source_i, subtract_prior=True):
        target_X = self.X_blocks[target_i]
        target_y = self.y_blocks[target_i]
        source_gp = self.gps[source_i]
        return local_pred_gradient_Xi_target(target_X, target_y, source_gp, subtract_prior=True)

    def _local_pred_gradient_Xi_source(self, target_i, source_i):
        target_X = self.X_blocks[target_i]
        target_y = self.y_blocks[target_i]
        source_gp = self.gps[source_i]
        return local_pred_gradient_Xi_source(target_X, target_y, source_gp)

    def llgrad_Xi_block(self, i, local=False):
        # derivatives of the total log-likelihood wrt each coordinate of the inputs for block i
        gp = self.gps[i]
        grad = gp.grad_ll_wrt_X()

        if local:
            sources = self.source_blocks(i)
            targets = self.target_blocks(i)
        else:
            sources = range(i)
            targets = range(i+1, self.n_blocks)

        for source_i in sources:
            g = self._local_pred_gradient_Xi_target(i, source_i, subtract_prior=True)
            grad += g

        for target_i in targets:
            grad += self._local_pred_gradient_Xi_source(target_i, i)

        return grad

    def llgrad_X(self, local=False):
        block_grads = [self.llgrad_Xi_block(i, local=local) for i in range(self.n_blocks)]
        grads = self.unscramble_Xblocks(block_grads)
        return grads

    def llgrad_Xi_block_parallel(self, i, p):
        # derivatives of the total log-likelihood wrt each coordinate of the inputs for block i
        gp = self.gps[i]
        grad = gp.grad_ll_wrt_X()

        target_X = self.X_blocks[i]
        target_y = self.y_blocks[i]
        args = [(target_X, target_y, self.gps[j]) for j in range(i)]
        sgrads = p.map(local_pred_gradient_Xi_target_args, args )

        source_gp = self.gps[i]
        args = [(self.X_blocks[j], self.y_blocks[j], source_gp) for j in range(i+1, self.n_blocks)]
        tgrads = p.map(local_pred_gradient_Xi_source_args, args )

        grad += np.sum(sgrads, axis=0) + np.sum(tgrads, axis=0)

        return grad

    def llgrad_X_parallel(self):
        p = Pool(4)
        block_grads = [self.llgrad_Xi_block_parallel(i, p) for i in range(self.n_blocks)]
        grads = self.unscramble_Xblocks(block_grads)
        return grads

    def _local_pred_ll(self, target_X, target_y, source_gp, subtract_prior=True):
        mean = source_gp.predict(target_X)
        cov = source_gp.covariance(target_X, include_obs=True)

        # p(target|source) under source model
        r = target_y - mean
        prec = np.linalg.inv(cov)
        ll = -.5 * np.dot(r.T, np.dot(prec, r))
        ll -= .5*np.log(np.linalg.det(cov))
        ll -= .5*len(r)*np.log(2*np.pi)


        if subtract_prior:
            # p(target) under source model
            prior_cov = source_gp.kernel(target_X, target_X, identical=True)
            prior_prec = np.linalg.inv(prior_cov)
            ll += .5 * np.dot(target_y.T, np.dot(prior_prec, target_y))
            ll += .5*np.log(np.linalg.det(prior_cov))
            ll += .5*len(target_y)*np.log(2*np.pi)

        return ll

    def likelihood_verbose(self):

        ll = 0
        for i in range(self.n_blocks):
            target_X = self.X_blocks[i]
            target_y = self.y_blocks[i]

            prior_lp = self.gps[i].log_likelihood()
            print "block", i
            print "  prior ll %.2f" % prior_lp

            correction = 0
            for j in range(i):
                source_gp = self.gps[j]
                mean = source_gp.predict(target_X)
                cov = source_gp.covariance(target_X, include_obs=True)

                prior_mean  = np.zeros(mean.shape)
                prior_cov = source_gp.kernel(target_X, target_X, identical=True)
                #prior_prec = np.linalg.inv(prior_cov)

                pred_rv = scipy.stats.multivariate_normal(mean, cov)
                prior_rv = scipy.stats.multivariate_normal(prior_mean, prior_cov)

                pred_ll = pred_rv.logpdf(target_y)
                prior_ll = prior_rv.logpdf(target_y)
                correction_j = pred_ll-prior_ll
                print "   from %d pred %.2f - prior %.2f = correction %.2f" % (j, pred_ll, prior_ll, correction_j)
                correction += correction_j
            print "  total correction %.2f, conditional logp %.2f" % (correction, prior_lp+correction)
            ll += prior_lp+correction

        print "OVERALL LP", ll

    def _local_pred_gradient(self, target_X, target_y, source_gp, subtract_prior=True):
        mean = source_gp.predict(target_X)
        cov = source_gp.covariance(target_X, include_obs=True)

        nparams = len(source_gp.ll_grad)
        llgrad = np.zeros((nparams,))

        r = target_y - mean
        prec = np.linalg.inv(cov)

        if subtract_prior:
            prior_cov = source_gp.kernel(target_X, target_X, identical=True)
            prior_prec = np.linalg.inv(prior_cov)

        for i_hparam in range(nparams):
            dmean, dcov = source_gp.grad_prediction(target_X, i_hparam)

            llgrad[i_hparam] = dgaussian(r, prec, dcov, dmean)

            if subtract_prior:
                dcov = source_gp.dKdi(target_X, target_X, i_hparam, identical=True)
                llgrad[i_hparam] -= dgaussian(target_y, prior_prec, dcov)

        return llgrad

    def llgrad_hparam_block(self, i):
        gp = self.gps[i]
        grad = gp.ll_grad.copy()
        nparams = len(grad)

        for target_i in range(i+1, self.n_blocks):
            target_X = self.X_blocks[target_i]
            target_y = self.y_blocks[target_i]
            pllgrad = self._local_pred_gradient(target_X, target_y, gp, subtract_prior=True)
            grad += pllgrad

        return ll, grad

    def ll_block(self, i, local=False):
        gp = self.gps[i]
        ll = gp.log_likelihood()

        if local:
            targets = self.target_blocks(i)
        else:
            targets = range(i+1, self.n_blocks)

        for target_i in targets:
            target_X = self.X_blocks[target_i]
            target_y = self.y_blocks[target_i]
            pll = self._local_pred_ll(target_X, target_y, gp, subtract_prior=True)
            ll += pll

        return ll

    def llgrad_hparam(self):
        grads = [self.llgrad_hparam_block(i) for i in range(self.n_blocks)]

        if self.tied_params:
            llgrad = np.sum(grads, axis=0)
        else:
            llgrad = np.concatenate(grads)

        return ll, llgrad

    def permute(self, block_perm=None):
        if block_perm is None:
            block_perm = np.random.permutation(self.n_blocks)

        X_blocks = []
        y_blocks = []
        gps = []
        block_centers = []
        cov_blocks = []
        cov_block_params = []

        for b in block_perm:
            X_blocks.append(self.X_blocks[b])
            y_blocks.append(self.y_blocks[b])
            gps.append(self.gps[b])
            block_centers.append(self.block_centers[b])
            cov_blocks.append(self.cov_blocks[b])

            try:
                cov_block_params.append(self.cov_block_params[b])
            except IndexError:
                continue

        return BCM(block_centers=block_centers, cov_block_params=cov_block_params, X_blocks=X_blocks, y_blocks=y_blocks, cov_blocks=cov_blocks, gps=gps, test_cov=self.test_cov)


    def stochastic_llgrad(self, samples=1, pseudo=True):
        ll = 0
        llgrad = None
        for i in range(samples):
            p = np.random.permutation(self.n_blocks)
            sll, sllgrad = self.llgrad(pseudo=pseudo, block_perm=p)
            ll += sll
            llgrad = sllgrad if llgrad is None else llgrad+sllgrad
        return ll/samples, llgrad/samples

    def cycle_pseudoll(self):
        ll = 0
        llgrad = None
        n = self.n_blocks
        p = np.arange(n)
        for i in range(n):
            p[-1], p[i] = p[i], p[-1]
            sll, sllgrad = self.llgrad(pseudo=True, block_perm=p)
            ll += sll
            llgrad = sllgrad if llgrad is None else llgrad+sllgrad
            p[-1], p[i] = p[i], p[-1]
        return ll/n, llgrad/n

    def likelihood(self, local=False):
        lls = [self.ll_block(i, local=local) for i in range(self.n_blocks)]
        ll = np.sum(lls)
        return ll

    def likelihood_naive(self, verbose=False):
        ll = self.gps[0].log_likelihood()
        for i in range(1, self.n_blocks):
            nv, pc = self.cov_blocks[i]
            m, c = self.predict_dist(self.X_blocks[i], source_gps = self.gps[:i], noise_var=nv, predict_cov=pc)
            llb = scipy.stats.multivariate_normal(mean=m, cov=c).logpdf(self.y_blocks[i])
            if verbose:
                print "block %d conditional ll %.2f vs prior %.2f" % (i, llb, self.gps[i].log_likelihood())
            ll += llb
        return ll

    def update_X(self, flatX, **kwargs):
        # assign new locations to the same blocks as the old (even if this mixes blocks in space)

        X_blocks = []

        d = self.X_blocks[0].shape[1]
        for idxs in self.block_idxs:
            xb = []
            for idx in idxs:
                xb.append(flatX[idx*d:(idx+1)*d])
            X_blocks.append(np.array(xb, dtype=np.float64))

        if isinstance(self, BCM):
            kwargs['test_cov'] = self.test_cov
        bcm = self.__class__(block_centers=self.block_centers,
                              cov_block_params = self.cov_block_params,
                              X_blocks = X_blocks,
                              y_blocks = self.y_blocks,
                              **kwargs)
        bcm.block_idxs = self.block_idxs

        return bcm


    def compute_neighbors(self, threshold=0.999):

        def source_blocks(i):
            ci = self.block_centers[i].reshape((1, -1))
            v = np.array([float(np.min(gp.variance(self.X_blocks[i], include_obs=False))) for gp in self.gps[:i]])
            prior_v = np.array([float(gp.kernel(ci, ci, identical=False)) for gp in self.gps[:i]])
            vmargin = v/prior_v
            relevant_blocks = vmargin < threshold
            return np.arange(len(vmargin))[relevant_blocks]

        def target_blocks(i):
            gp = self.gps[i]
            v = np.array([float(np.min(gp.variance(xb, include_obs=False))) for xb in self.X_blocks[i+1:]])

            ci = self.block_centers[i].reshape((1, -1))
            prior_v = float(gp.kernel(ci, ci, identical=False))
            vmargin = v/prior_v
            relevant_blocks = vmargin < threshold
            return np.arange(i+1, self.n_blocks)[relevant_blocks]

        self.neighbors = []
        for i in range(self.n_blocks):
            sources = source_blocks(i)
            targets = target_blocks(i)
            self.neighbors.append((sources, targets))

    def source_blocks(self, i):
        return self.neighbors[i][0]

    def target_blocks(self, i):
        return self.neighbors[i][1]

class MultiGPLVM(object):

    def __init__(self, X, Y, cov_block_params, tied_hparams=True, bcm=False, **kwargs):
        self.lgps = []
        for y in Y.T:
            if bcm:
                lgp = BCM(X=X, y=y, cov_block_params=cov_block_params, **kwargs)
            else:
                lgp = LocalGPs(X=X, y=y, cov_block_params=cov_block_params, **kwargs)
            self.lgps.append(lgp)

        self.bcm = bcm
        self.tied_hparams = tied_hparams

    def llgrad_hparam(self, **kwargs):
        llgrads = [lgp.llgrad_hparam(**kwargs) for lgp in self.lgps]
        lls, grads = zip(*llgrads)
        ll = np.sum(lls)
        if self.tied_hparams:
            grad = np.sum(grads, axis=0)
        else:
            grad = np.concatenate(grads)
        return ll, grad

    def llgrad_X(self, **kwargs):
        grads = [lgp.llgrad_X(**kwargs) for lgp in self.lgps]
        grad = np.sum(grads, axis=0)
        return grad

    def likelihood(self, **kwargs):
        return np.sum([lgp.likelihood(**kwargs) for lgp in self.lgps])

    def update_X(self, xx, **kwargs):
        self.lgps = [lgp.update_X(xx, **kwargs) for lgp in self.lgps]



def local_pred_gradient_Xi_target_args(args):
    return local_pred_gradient_Xi_target(*args)

def local_pred_gradient_Xi_target(target_X, target_y, source_gp, subtract_prior=True):
    mean = source_gp.predict(target_X)
    cov = source_gp.covariance(target_X, include_obs=True)
    r = target_y.flatten() - mean.flatten()
    prec = np.linalg.inv(cov)
    alpha = np.dot(prec, r)

    prior_cov = source_gp.kernel(target_X, target_X, identical=True)
    prior_prec = np.linalg.inv(prior_cov)
    prior_alpha = np.dot(prior_prec, target_y)


    # can be shared over p, i
    Kstar = source_gp.kernel(target_X, source_gp.X)
    tmp = np.dot(source_gp.Kinv, Kstar.T)
    #Kss = prior_cov #self.kernel(X1, X1, identical=True)



    n, d = target_X.shape
    llgrad = np.zeros((n, d))
    for p in range(n):
        for i in range(d):


            dKss = source_gp.dKdx(target_X, p, i, return_vec=True)

            # this is really the p'th row of dKstar, all other rows are zero
            dKstar = source_gp.dKdx(target_X[p:p+1, :], 0, i, X2=source_gp.X)
            dm = np.dot(dKstar, source_gp.alpha_r)


            #dm = np.dot(dKstar, bcm.alpha_r)

            dqf = np.dot(dKstar, tmp)
            #dc = dKss - dqf - dqf.T
            dc = np.asarray(dKss - dqf).flatten()
            dc[p] *= 2

            llgrad[p, i] = dgaussian_rank1(r, alpha, prec, dc, p, dm)

            #dm1, dc1 = source_gp.grad_prediction_wrt_target_x(target_X, p, i)
            #llgrad_old = dgaussian(r, prec, dc1, dm1)

            #print p, i, llgrad[p, i], llgrad_old

            #print "setting %d, %d = %f" % (p, i, llgrad[p, i])


            if subtract_prior:
            # compute the llgrad from the message from the source to target, by
            # subtracting the gradient of p_source(target)
            # (the target's prior likelihood in the source model)
                dcv = source_gp.dKdx(target_X, p, i, return_vec=True)
                plp = dgaussian_rank1(target_y, prior_alpha, prior_prec, dcv, p)
                llgrad[p, i] -= plp
                #print "subtracting %f from %d, %d = %f" % (plp, p, i, llgrad[p, i])

    return llgrad

def local_pred_gradient_Xi_source_args(args):
    return local_pred_gradient_Xi_source(*args)

def shared_local_pred_gradient(target_X, source_gp):
    # how much of this can I compute without knowing the source or target y, r, alpha?

    n, d = source_gp.X.shape
    llgrad = np.zeros((n, d))
    Kinv = source_gp.Kinv

    #alpha_source = source_gp.alpha_r

    #mean = source_gp.predict(target_X)
    cov = source_gp.covariance(target_X, include_obs=True)
    #r = target_y.flatten() - mean.flatten()
    prec = np.linalg.inv(cov)
    #alpha_target = np.dot(prec, r)

    # can share Kstar over different vals of p, i
    Kstar = source_gp.kernel(target_X, source_gp.X)
    Kstar_Kyinv = np.dot(Kstar, Kinv)

    m1 = np.asarray(np.dot(prec, Kstar_Kyinv))
    m2 = np.asarray(np.dot(Kstar_Kyinv.T, m1))

    for p in range(n):
        for i in range(d):

            # this gives the p'th column of dKstar, all others are zero
            dKstar = np.asarray(source_gp.dKdx(source_gp.X[p:p+1, :], 0, i, target_X)).flatten()

            # dKy has one nonzero row and column
            dKy = source_gp.dKdx(source_gp.X, p, i, return_vec=True)

            #dK_alpha = alpha_source[p] * dKy
            #dK_alpha[p] = np.dot(dKy, alpha_source)

            #dKstar_alpha = np.asarray(alpha_source[p] * dKstar).flatten()

            gammaprime = np.asarray(np.dot(Kstar_Kyinv, dK_alpha)).flatten()
            dm = dKstar_alpha - gammaprime
            dll_dmean = np.dot(r.T, np.dot(prec, dm))




            # let C = cov as computed above
            # dll_dcov = r' C^-1 dCdx C^-1 r + a trace term
            #          = alpha' dCdx alpha (ignoring trace term)
            # where dCdx = (Kstar Kinv dK Kinv Kstar + dKstar Kinv Kstar + Kstar Kinv dKstar)
            # Distributing alpha over dCdx, we get the first term below corresponding
            # to the first term above, and the second term below collapses the second
            # two terms above (which are symmetric).
            Kyinv_KstarT_talpha = np.asarray(np.dot(Kstar_Kyinv.T, alpha_target)).flatten()
            tmp = Kyinv_KstarT_talpha[p] * dKy
            tmp[p] = np.dot(dKy, Kyinv_KstarT_talpha)
            dll_dcov_term1 = -np.dot(Kyinv_KstarT_talpha.T, tmp)

            # this is really a vector whose p'th entry is this scalar
            dKstarT_talpha = np.dot(dKstar.T, alpha_target)
            dll_dcov_term2 = 2*Kyinv_KstarT_talpha[p] * dKstarT_talpha

            # The trace term is tr(C^-1 dCdx) which again we can distribute over the
            # terms of dCdx, and using the cyclic property of trace we can rewrite as
            # the hypothetical computation
            #    cov_tr_factor = 2*np.sum(m1 * dKstar) - np.sum(m2 * dKy)
            # where m1 and m2 are as precomputed above. However we have to do a bit more
            # work to exploit the fact that we have dKstar and dKy in sparse form.
            cov_tr_factor = -2*np.sum(m1[:,p] * dKstar) + 2*np.sum(m2[:,p] * dKy)
            dll_dcov = -.5* (dll_dcov_term1 + dll_dcov_term2 + cov_tr_factor)

            llgrad[p,i] = dll_dcov + dll_dmean

            # ORIGINAL
            #dm2, dc2 = source_gp.grad_prediction_wrt_source_x(target_X, p, i)
            #llgrad2 = dgaussian(r, prec, dc2, dm2)

            #print llgrad[p,i], llgrad2


    return llgrad




def local_pred_gradient_Xi_source(target_X, target_y, source_gp):
    n, d = source_gp.X.shape
    llgrad = np.zeros((n, d))
    Kinv = source_gp.Kinv
    alpha_source = source_gp.alpha_r

    mean = source_gp.predict(target_X)
    cov = source_gp.covariance(target_X, include_obs=True)
    r = target_y.flatten() - mean.flatten()
    prec = np.linalg.inv(cov)
    alpha_target = np.dot(prec, r)

    # can share Kstar over different vals of p, i
    Kstar = source_gp.kernel(target_X, source_gp.X)
    Kstar_Kyinv = np.dot(Kstar, Kinv)

    m1 = np.asarray(np.dot(prec, Kstar_Kyinv))
    m2 = np.asarray(np.dot(Kstar_Kyinv.T, m1))

    for p in range(n):
        for i in range(d):

            # this gives the p'th column of dKstar, all others are zero
            dKstar = np.asarray(source_gp.dKdx(source_gp.X[p:p+1, :], 0, i, target_X)).flatten()

            # dKy has one nonzero row and column
            dKy = source_gp.dKdx(source_gp.X, p, i, return_vec=True)

            dK_alpha = alpha_source[p] * dKy
            dK_alpha[p] = np.dot(dKy, alpha_source)

            dKstar_alpha = np.asarray(alpha_source[p] * dKstar).flatten()

            gammaprime = np.asarray(np.dot(Kstar_Kyinv, dK_alpha)).flatten()
            dm = dKstar_alpha - gammaprime
            dll_dmean = np.dot(r.T, np.dot(prec, dm))


            # let C = cov as computed above
            # dll_dcov = r' C^-1 dCdx C^-1 r + a trace term
            #          = alpha' dCdx alpha (ignoring trace term)
            # where dCdx = (Kstar Kinv dK Kinv Kstar + dKstar Kinv Kstar + Kstar Kinv dKstar)
            # Distributing alpha over dCdx, we get the first term below corresponding
            # to the first term above, and the second term below collapses the second
            # two terms above (which are symmetric).
            Kyinv_KstarT_talpha = np.asarray(np.dot(Kstar_Kyinv.T, alpha_target)).flatten()
            tmp = Kyinv_KstarT_talpha[p] * dKy
            tmp[p] = np.dot(dKy, Kyinv_KstarT_talpha)
            dll_dcov_term1 = -np.dot(Kyinv_KstarT_talpha.T, tmp)

            # this is really a vector whose p'th entry is this scalar
            dKstarT_talpha = np.dot(dKstar.T, alpha_target)
            dll_dcov_term2 = 2*Kyinv_KstarT_talpha[p] * dKstarT_talpha

            # The trace term is tr(C^-1 dCdx) which again we can distribute over the
            # terms of dCdx, and using the cyclic property of trace we can rewrite as
            # the hypothetical computation
            #    cov_tr_factor = 2*np.sum(m1 * dKstar) - np.sum(m2 * dKy)
            # where m1 and m2 are as precomputed above. However we have to do a bit more
            # work to exploit the fact that we have dKstar and dKy in sparse form.
            cov_tr_factor = -2*np.sum(m1[:,p] * dKstar) + 2*np.sum(m2[:,p] * dKy)
            dll_dcov = -.5* (dll_dcov_term1 + dll_dcov_term2 + cov_tr_factor)

            llgrad[p,i] = dll_dcov + dll_dmean

            # ORIGINAL
            #dm2, dc2 = source_gp.grad_prediction_wrt_source_x(target_X, p, i)
            #llgrad2 = dgaussian(r, prec, dc2, dm2)

            #print llgrad[p,i], llgrad2


    return llgrad
