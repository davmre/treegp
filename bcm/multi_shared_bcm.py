import numpy as np
import scipy.stats

from collections import defaultdict
from multiprocessing import Pool

from treegp.gp import GP, GPCov, mcov, dgaussian, dgaussian_rank1

from treegp.cover_tree import VectorTree
import pyublas


class Blocker(object):

    def __init__(self, block_centers):
        self.block_centers = block_centers
        self.n_blocks = len(block_centers)

    def get_block(self, X_new):
        dists = [np.linalg.norm(X_new - center) for center in self.block_centers]
        return np.argmin(dists)

    def sort_by_block(self, X, Y=None, YY=None):
        if Y is not None:
            assert(Y.shape[0] == X.shape[0])
        elif YY is not None:
            assert(YY.shape[0] == X.shape[0])
            assert(YY.shape[1] == X.shape[0])

        n = X.shape[0]
        blocks = np.array([self.get_block(xp) for xp in X])
        idxs = np.arange(n)
        perm = np.asarray(sorted(idxs, key = lambda i : blocks[i]))
        sorted_X = X[perm]
        sorted_blocks = blocks[perm]
        block_idxs = [idxs[sorted_blocks==block] for block in np.arange(self.n_blocks)]
        block_boundaries = [ (np.min(bi), np.max(bi)+1)  for bi in block_idxs  ]

        if Y is not None:
            sorted_Y = Y[perm]
            return sorted_X, sorted_Y, perm, block_boundaries
        elif YY is not None:
            tmp = YY[perm]
            sorted_YY = tmp[:, perm]
            return sorted_X, sorted_YY, perm, block_boundaries
        else:
            return sorted_X, perm, block_boundaries

def sample_synthetic(seed=1, n=400, xd=2, yd=10, lscale=0.1, noise_var=0.01):
    # sample data from the prior
    np.random.seed(seed)
    X = np.random.rand(n, xd)

    cov = GPCov(wfn_params=[1.0], dfn_params=[lscale, lscale], dfn_str="euclidean", wfn_str="se")
    KK = mcov(X, cov, noise_var)

    y = scipy.stats.multivariate_normal(mean=np.zeros((X.shape[0],)), cov=KK).rvs(yd).T.reshape((-1, yd))

    return X, y, cov

def sample_synthetic_bcm_new(seed=1, n=400, xd=2, yd=10, lscale=0.1, noise_var=0.01, blocker=None):
    # sample data from the prior
    np.random.seed(seed)
    X = np.random.rand(n, xd)

    SX, perm, block_boundaries = blocker.sort_by_block(X)
    cov = GPCov(wfn_params=[1.0], dfn_params=[lscale, lscale], dfn_str="euclidean", wfn_str="se")

    Y = []
    Ks = []
    Js = []
    Kstar_precs = [] # (i,j) is K_{ij} * K_jj^{-1}
    pred_precs = [] # (i,j) is inv(K_ii - K_{ij} K_jj^{-1} K_{ji})
    combined_covs = [] # [i] is the covariance of p(Y_i |_B Y_:i), i.e. the covariance of the BCM conditional
    combined_chols = [] # cholesky decompositions of combined_covs
    for i, (i_start, i_end) in enumerate(block_boundaries):
        Xi = SX[i_start:i_end, :]
        Ki = mcov(Xi, cov, noise_var)
        Ji = np.linalg.inv(Ki)

        Ks.append(Ki)
        Js.append(Ji)

        precs = []
        Ksprecs_j = []
        pprecs_j = []
        for j, (j_start, j_end) in enumerate(block_boundaries[:i]):
            Xj = SX[j_start:j_end, :]
            Kj = Ks[j]
            Jj = Js[j]
            Kstar = mcov(Xi, cov, noise_var, X2=Xj)
            Kstar_prec = np.dot(Kstar, Jj)
            pred_cov = Ki - np.dot(Kstar_prec, Kstar.T)
            pred_prec = np.linalg.inv(pred_cov)
            message_prec = pred_prec - Ji
            precs.append(message_prec)

            if np.max(np.abs(Kstar_prec)) < 1e-3:
                pprecs_j.append(None)
                Ksprecs_j.append(None)
            else:
                pprecs_j.append(pred_prec)
                Ksprecs_j.append(Kstar_prec)

        pred_precs.append(pprecs_j)
        Kstar_precs.append(Ksprecs_j)

        combined_prec = np.sum(precs, axis=0) + Ji
        combined_cov = np.linalg.inv(combined_prec)
        combined_chol = np.linalg.cholesky(combined_cov)
        combined_covs.append(combined_cov)
        combined_chols.append(combined_chol)
        print "precomputed covs for block", i
    Y = []
    for d in range(yd):
        yis = []
        for i, (i_start, i_end) in enumerate(block_boundaries):
            means = [np.zeros((i_end-i_start,))]
            for j, (j_start, j_end) in enumerate(block_boundaries[:i]):
                if Kstar_precs[i][j] is not None:
                    pred_mean = np.dot(Kstar_precs[i][j], yis[j])
                    weighted_mean = np.dot(pred_precs[i][j], pred_mean)
                    means.append(weighted_mean)
            mean = np.dot(combined_covs[i],  np.sum(means, axis=0))

            yi = np.dot(combined_chols[i], np.random.randn(i_end-i_start)) + mean
            yis.append(yi)
        print "sampled y", d
        Yd = np.concatenate(yis).reshape((-1, 1))
        Y.append(Yd)
    Y = np.hstack(Y)

    perm = np.random.permutation(n)
    SX = SX[perm]
    Y = Y[perm]

    return SX, Y, cov


class MultiSharedBCM(object):

    def __init__(self, X, Y, block_boundaries, cov, noise_var, kernelized=False, dy=None, neighbor_threshold=1e-3):
        self.X = X

        if kernelized:
            self.kernelized = True
            self.YY = Y
            assert(dy is not None)
            self.dy = dy
        else:
            self.kernelized = False
            self.Y = Y
        self.block_boundaries = block_boundaries
        self.n_blocks = len(block_boundaries)

        self.cov = cov
        self.noise_var = noise_var
        dummy_X = np.array([[0.0,] * self.X.shape[1],], dtype=float)
        self.predict_tree = VectorTree(dummy_X, 1, cov.dfn_str, cov.dfn_params, cov.wfn_str, cov.wfn_params)

        self.compute_neighbors(threshold=neighbor_threshold)
        self.neighbor_threshold = neighbor_threshold

    def compute_neighbors(self, threshold=1e-3):
        neighbor_count = defaultdict(int)
        neighbors = []
        for i in range(self.n_blocks):
            i_start, i_end = self.block_boundaries[i]
            Xi = self.X[i_start:i_end]
            ni = Xi.shape[0]
            for j in range(i):
                j_start, j_end = self.block_boundaries[j]
                Xj = self.X[j_start:j_end]
                Kij = self.kernel(Xi, X2=Xj)
                maxk = np.max(np.abs(Kij))
                if maxk > threshold:
                    neighbors.append((i,j))
                    neighbor_count[i] += 1
                    neighbor_count[j] += 1
        self.neighbor_count = neighbor_count
        self.neighbors = neighbors

    def update_X(self, new_X, recompute_neighbors=False):
        self.X = new_X
        if recompute_neighbors:
            self.compute_neighbors(threshold=self.neighbor_threshold)

    def llgrad_blocked(self, parallel=False, **kwargs):
        """
        Compute likelihood under a model with blocked local GPs (no pairwise corrections)
        """
        if parallel:
            pool = Pool(processes=4)
            unary_args = [(kwargs, self, i) for i in range(self.n_blocks)]
            unaries = pool.map(llgrad_unary_shim, unary_args)
            pool.close()
            pool.join()
        else:
            unaries = [self.llgrad_unary(i, **kwargs) for i in range(self.n_blocks)]

        unary_lls, unary_grads = zip(*unaries)

        ll = np.sum(unary_lls)

        if "grad_X" in kwargs and kwargs['grad_X']:
            grads = np.zeros(self.X.shape)
        else:
            grads = np.zeros((0, 0))

        for i in range(self.n_blocks):
            i_start, i_end = self.block_boundaries[i]
            grads[i_start:i_end, :] += unary_grads[i]

        return ll, grads

    def llgrad(self, parallel=False, local=True, **kwargs):
        # overall likelihood is the pairwise potentials for all (unordered) pairs,
        # where each block is involved in (n-1) pairs. So we subtract each unary potential n-1 times.
        # Then finally each unary potential gets added in once.

        if local:
            neighbors = self.neighbors
            neighbor_count = self.neighbor_count
        else:
            neighbors = [(i,j) for i in range(self.n_blocks) for j in range(i)]
            neighbor_count = dict([(i, self.n_blocks-1) for i in range(self.n_blocks)])

        if parallel:
            pool = Pool(processes=4)
            unary_args = [(kwargs, self, i) for i in range(self.n_blocks)]
            unaries = pool.map(llgrad_unary_shim, unary_args)

            pair_args = [(kwargs, self, i, j) for (i,j) in neighbors]
            pairs = pool.map(llgrad_joint_shim, pair_args)
            pool.close()
            pool.join()
        else:
            t0 = time.time()
            unaries = [self.llgrad_unary(i, **kwargs) for i in range(self.n_blocks)]
            t1 = time.time()
            print self.n_blocks, "unaries in", t1-t0, "seconds"
            pairs = [self.llgrad_joint(i, j, **kwargs) for (i,j) in neighbors]
            t2 = time.time()
            print len(neighbors), "pairs in", t2-t1, "seconds"

        t0 = time.time()

        unary_lls, unary_grads = zip(*unaries)
        if len(pairs) > 0:
            pair_lls, pair_grads = zip(*pairs)
        else:
            pair_lls = []
            pair_grads = []

        ll = np.sum(pair_lls)
        ll -= np.sum([(neighbor_count[i]-1)*ull for (i, ull) in enumerate(unary_lls) ])

        if "grad_X" in kwargs and kwargs['grad_X']:
            grads = np.zeros(self.X.shape)
        else:
            grads = np.zeros((0, 0))

        pair_idx = 0
        for i in range(self.n_blocks):
            i_start, i_end = self.block_boundaries[i]
            grads[i_start:i_end, :] -= (neighbor_count[i]-1)*unary_grads[i]

        for pair_idx, (i,j) in enumerate(neighbors):
            i_start, i_end = self.block_boundaries[i]
            j_start, j_end = self.block_boundaries[j]
            ni = i_end-i_start
            grads[i_start:i_end] += pair_grads[pair_idx][:ni]
            grads[j_start:j_end] += pair_grads[pair_idx][ni:]

        t1 = time.time()


        return ll, grads


    def llgrad_unary(self, i, **kwargs):
        i_start, i_end = self.block_boundaries[i]
        X = self.X[i_start:i_end]

        if self.kernelized:
            YY = self.YY[i_start:i_end, i_start:i_end]
            return self.gaussian_llgrad_kernel(X, YY, dy=self.dy, **kwargs)
        else:
            Y = self.Y[i_start:i_end, :]
            return self.gaussian_llgrad(X, Y, **kwargs)

    def llgrad_joint(self, i, j, **kwargs):
        i_start, i_end = self.block_boundaries[i]
        j_start, j_end = self.block_boundaries[j]
        Xi = self.X[i_start:i_end]
        Xj = self.X[j_start:j_end]

        ni = Xi.shape[0]
        nj = Xj.shape[0]
        X = np.vstack([Xi, Xj])

        if self.kernelized:
            YY = np.empty((ni+nj, ni+nj))
            YY[:ni, :ni] = self.YY[i_start:i_end, i_start:i_end]
            YY[ni:, ni:] = self.YY[j_start:j_end, j_start:j_end]
            YY[:ni, ni:] = self.YY[i_start:i_end, j_start:j_end]
            YY[ni:, :ni]  = YY[:ni, ni:].T
            return self.gaussian_llgrad_kernel(X, YY, dy=self.dy, **kwargs)
        else:
            Yi = self.Y[i_start:i_end, :]
            Yj = self.Y[j_start:j_end, :]
            Y = np.vstack([Yi, Yj])
            return self.gaussian_llgrad(X, Y, **kwargs)


    def kernel(self, X, X2=None):
        if X2 is None:
            n = X.shape[0]
            K = self.predict_tree.kernel_matrix(X, X, False)
            K += np.eye(n) * self.noise_var
        else:
            K = self.predict_tree.kernel_matrix(X, X2, False)
        return K

    def dKdx(self, X, p, i, return_vec=False):
        # derivative of kernel(X1, X2) wrt i'th coordinate of p'th point in X1.
        if return_vec:
            dKv = self.predict_tree.kernel_deriv_wrt_xi_row(X, p, i)
            dKv[p] = 0
            return dKv
        else:
            dK = self.predict_tree.kernel_deriv_wrt_xi(X, X, p, i)
            dK[p,p] = 0
            dK = dK + dK.T
            return dK

    def gaussian_llgrad(self, X, Y, grad_X = False):

        t0 = time.time()
        n, dx = X.shape
        dy = Y.shape[1]

        K = mcov(X, self.cov, self.noise_var)
        prec = np.linalg.inv(K)
        Alpha = np.dot(prec, Y)

        ll = -.5 * np.sum(Y*Alpha)
        ll += -.5 * dy * np.linalg.slogdet(K)[1]
        ll += -.5 * dy * n * np.log(2*np.pi)

        if not grad_X:
            return ll, np.array(())

        llgrad = np.zeros((n, dx))
        for p in range(n):
            for i in range(dx):
                dll = 0
                dcv = self.dKdx(X, p, i, return_vec=True)
                #t1 = -np.outer(prec[p,:], dcv)
                #t1[:, p] = -np.dot(prec, dcv)
                #dll_dcov = .5*ny*np.trace(t1)

                dll = -dy * np.dot(prec[p,:], dcv)

                for j in range(dy):
                    alpha = Alpha[:,j]
                    dK_alpha = dcv * alpha[p]
                    dK_alpha[p] = np.dot(dcv, alpha)
                    dll_dcov = .5*np.dot(alpha, dK_alpha)
                    dll += dll_dcov

                llgrad[p,i] = dll
        t1 = time.time()
        #print "llgrad %d pts %.4s" % (n, t1-t0)
        return ll, llgrad

    def gaussian_llgrad_kernel(self, X, YY, dy=None, grad_X=False):
        n, dx = X.shape
        if dy is None:
            dy = self.dy

        K = mcov(X, self.cov, self.noise_var)
        Kinv = np.linalg.inv(K)
        prec = Kinv

        KYYK = np.dot(np.dot(Kinv, YY), Kinv)

        ll =  -.5 * np.sum(Kinv * YY)
        ll += -.5 * dy * np.linalg.slogdet(K)[1]
        ll += -.5 * dy * n * np.log(2*np.pi)
        if not grad_X:
            return ll, np.array(())

        llgrad = np.zeros((n, dx))
        for p in range(n):
            for i in range(dx):
                #dcv_full = self.dKdx(X, p, i)
                #dll = -.5*np.sum(KYYK * dcv_full)
                dcv = self.dKdx(X, p, i, return_vec=True)
                dll = np.dot(KYYK[p,:], dcv)

                dll += -dy * np.dot(prec[p,:], dcv)

                llgrad[p,i] = dll

                #t1 = -np.outer(prec[p,:], dcov_v)
                #t1[:, p] = -np.dot(prec, dcov_v)

        return ll, llgrad

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['predict_tree']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        dummy_X = np.array([[0.0,] * self.X.shape[1],], dtype=float)
        self.predict_tree = VectorTree(dummy_X, 1, self.cov.dfn_str, self.cov.dfn_params, self.cov.wfn_str, self.cov.wfn_params)

from multiprocessing import Pool
import time

def llgrad_unary_shim(arg):
    return MultiSharedBCM.llgrad_unary(*arg[1:], **arg[0])

def llgrad_joint_shim(arg):
    return MultiSharedBCM.llgrad_joint(*arg[1:], **arg[0])
