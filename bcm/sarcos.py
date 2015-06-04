from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic
from treegp.bcm.local_regression import BCM

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from treegp.util import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import scipy.io
import time
import os
import sys
import cPickle as pickle
import argparse


def load_sarcos():
    train_fname = "/home/dmoore/sarcos/sarcos_inv.mat"
    test_fname = "/home/dmoore/sarcos/sarcos_inv_test.mat"
    train_X = scipy.io.loadmat(train_fname)['sarcos_inv']
    train_y = train_X[:, 21]
    train_X = train_X[:, :21]

    train_y -= np.mean(train_y)
    train_y /= np.std(train_y)
    return train_X, train_y

def cluster_rpc((X, y), target_size):
    n = X.shape[0]
    if n < target_size:
        return [(X, y),]

    x1 = X[np.random.randint(n), :]
    x2 = x1
    while (x2==x1).all():
        x2 = X[np.random.randint(n), :]

    # what's the projection of x3 onto (x1-x2)?
    # imagine that x2 is the origin, so it's just x3 onto x1.
    # This is x1 * <x3, x1>/||x1||
    cx1 = x1 - x2
    nx1 = cx1 / np.linalg.norm(cx1)
    alphas = [ np.dot(xi-x2, nx1)  for xi in X]
    median = np.median(alphas)
    C1 = (X[alphas < median], y[alphas < median])
    C2 = (X[alphas >= median], y[alphas >= median])

    L1 = cluster_rpc(C1, target_size=target_size)
    L2 = cluster_rpc(C2, target_size=target_size)
    return L1 + L2

def sort_by_cluster(clusters):
    Xs, ys = zip(*clusters)
    SX = np.vstack(Xs)
    SY = np.concatenate(ys).reshape((-1, 1))
    block_boundaries = []
    n = 0
    for (X,y) in clusters:
        cn = X.shape[0]
        block_boundaries.append((n, n+cn))
        n += cn
    return SX, SY, block_boundaries


def optimize_cov(d, mbcm, seed, maxsec=36000):
    means = np.ones((23,), dtype=float)
    means[0] = -4.0
    means[1] = 0.5
    means[2:] = 2.0

    np.random.seed(seed)
    c0 = np.random.randn(23) + means
    C0 = c0.reshape((1, -1))

    def cov_prior(c):
        std = 3
        r = (c-means)/std
        ll = -.5*np.sum( r**2)- .5 *len(c) * np.log(2*np.pi*std**2)
        lderiv = -(c-means)/(std**2)
        return ll, lderiv

    sstep = [0,]
    t0 = time.time()

    def lgpllgrad(c):

        FC = np.exp(c.reshape(C0.shape))
        mbcm.update_covs(FC)
        np.save(os.path.join(d, "step_%05d_cov.npy" % sstep[0]), FC)

        ll, gX, gC = mbcm.llgrad(local=True, grad_X=False,
                                 grad_cov=True,
                                 parallel=False)

        prior_ll, prior_grad = cov_prior(c)
        ll += prior_ll
        gC = (gC * FC).flatten() + prior_grad

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        print FC

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            print "killing, final log covs", c.reshape(C0.shape)
            raise OutOfTimeError

        return -ll, -gC

    bounds = [(-7, 7)]*len(c0)
    r = scipy.optimize.minimize(lgpllgrad, c0, jac=True, method="l-bfgs-b", bounds=bounds)

def sarcos_exp_dir(seed, block_size, thresh, npts):
    base_dir = "sarcos_experiments"
    run_name = "%d_%d_%.4f_%d" % (seed, block_size, thresh, npts)
    d =  os.path.join(base_dir, run_name)
    mkdir_p(d)
    return d

def run_sarcos(seed, block_size, thresh, npts=None, maxsec=3600):
    d = sarcos_exp_dir(seed, block_size, thresh, npts)

    tX, ty = load_sarcos()
    n = len(ty)

    np.random.seed(seed)
    npts = n if (npts is None or npts <= 0) else npts
    p = np.random.permutation(n)[:npts]
    tX, ty = tX[p], ty[p]

    np.random.seed(seed)
    CC = cluster_rpc((tX, ty), target_size=block_size)
    SX, SY, block_boundaries = sort_by_cluster(CC)

    np.save(os.path.join(d, "SX.npy"), SX)
    np.save(os.path.join(d, "SY.npy"), SY)
    np.save(os.path.join(d, "blocks.npy"), np.array(block_boundaries))

    nv = 0.001
    cov = GPCov(wfn_params=[1.0], dfn_params=np.ones((1, SX.shape[1]))*10, dfn_str="euclidean", wfn_str="se")

    mbcm = MultiSharedBCM(SX, SY, block_boundaries, cov, nv,
                          dy=1, neighbor_threshold=thresh,
                          nonstationary=False,
                          nonstationary_prec=False)

    with open(os.path.join(d, "neighbors.txt"), 'w') as f:
        for i in range(mbcm.n_blocks):
            f.write("%d: %d neighbors\n" %(i, mbcm.neighbor_count[i]))
        mean_neighbor_count = np.mean(mbcm.neighbor_count.values())
        total_pairs = len(mbcm.neighbors)
        f.write("total pairs %d, average neighbor count %.3f\n" % (total_pairs, mean_neighbor_count))

    optimize_cov(d, mbcm, seed, maxsec)

def main():

    parser = argparse.ArgumentParser(description='bcmopt')
    parser.add_argument('--npts', dest='npts', default=-1, type=int)
    parser.add_argument('--block_pts', dest='block_pts', default=300, type=int)
    parser.add_argument('--threshold', dest='threshold', default=0.0, type=float)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int)
    parser.add_argument('--analyze', dest='analyze', default=False, action="store_true")
    args = parser.parse_args()
    run_sarcos(args.seed, args.block_pts, args.threshold, args.npts, args.maxsec)

if __name__ == "__main__":
    main()
