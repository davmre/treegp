from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic
from treegp.bcm.local_regression import BCM
from treegp.bcm.bcmopt import OutOfTimeError, load_log, cluster_rpc, sort_by_cluster

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
    base = os.path.join(os.environ["HOME"], "sarcos")
    train_fname = os.path.join(base, "sarcos_inv.mat")
    test_fname = os.path.join(base, "sarcos_inv_test.mat")

    train_X = scipy.io.loadmat(train_fname)['sarcos_inv']
    train_y = train_X[:, 21]
    train_X = train_X[:, :21]


    test_X = scipy.io.loadmat(test_fname)['sarcos_inv_test']
    test_y = test_X[:, 21]
    test_X = test_X[:, :21]
    
    # remember to do this when the time comes
    test_y -= np.mean(train_y)
    test_y /= np.std(train_y)

    train_y -= np.mean(train_y)
    train_y /= np.std(train_y)

    return train_X, train_y, test_X, test_y


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

    rfname = os.path.join(d, "log.txt")
    results = open(rfname, 'w')
    print "writing log to", rfname

    cfname = os.path.join(d, "covs.txt")
    covf = open(cfname, 'w')
    print "writing covs to", cfname
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

        s =  "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        print s
        results.write(s + "\n")
        results.flush()
        
        covf.write("%d %s\n" % (sstep[0], FC))

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            print "killing, final log covs", c.reshape(C0.shape)
            raise OutOfTimeError

        return -ll, -gC

    bounds = [(-7, 7)]*len(c0)
    r = scipy.optimize.minimize(lgpllgrad, c0, jac=True, method="l-bfgs-b", bounds=bounds)

    results.close()
    covf.close()

def sarcos_exp_dir(seed, block_size, thresh, npts):
    base_dir = "sarcos_experiments"
    run_name = "%d_%d_%.4f_%d" % (seed, block_size, thresh, npts)
    d =  os.path.join(base_dir, run_name)
    mkdir_p(d)
    return d

def run_sarcos(seed, block_size, thresh, npts=None, maxsec=3600):
    d = sarcos_exp_dir(seed, block_size, thresh, npts)

    tX, ty, _, _ = load_sarcos()
    n = len(ty)

    np.random.seed(seed)
    npts = n if (npts is None or npts <= 0) else npts
    p = np.random.permutation(n)[:npts]
    tX, ty = tX[p], ty[p]
    ty = ty.reshape((-1, 1))

    np.random.seed(seed)
    CC = cluster_rpc((tX, ty), target_size=block_size)
    SX, SY, block_boundaries = sort_by_cluster(CC)
    SY = SY.flatten()

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


def eval_predict(seed, block_size, thresh, npts=None, 
                 predict_from_fullX=True, ntest=-1):

    # for a given training set and cov, evaluate on test data

    d = sarcos_exp_dir(seed, block_size, thresh, npts)
    X, y, X_test, y_test = load_sarcos()

    if predict_from_fullX:
        np.random.seed(seed)
        CC = cluster_rpc((X, y, np.arange(self.SX.shape[0])), target_size=block_size)
        SX, SY, perm, block_boundaries = sort_by_cluster(CC)        
    else:
        SX = np.load(os.path.join(d, "SX.npy"))
        SY = np.load(os.path.join(d, "SY.npy"))
        block_boundaries = [row for row in np.load(os.path.join(d, "blocks.npy"))]

    # create using the "standard" cov so that the neighbors end up the same as predicted
    nv = 0.001
    cov = GPCov(wfn_params=[1.0], dfn_params=np.ones((1, SX.shape[1]))*10, dfn_str="euclidean", wfn_str="se")
    mbcm = MultiSharedBCM(SX, SY, block_boundaries, cov, nv,
                          dy=1, neighbor_threshold=0.00,
                          nonstationary=False,
                          nonstationary_prec=False)

    steps, times, lls = load_log(d)
    best_step = np.argmax(lls)
    FC = np.load(os.path.join(d, "step_%05d_cov.npy" % best_step))
    test_cov = GPCov(wfn_params=[FC[0,1]], dfn_params=FC[0,2:], dfn_str="euclidean", wfn_str="se")
    test_nv = FC[0,0]
    
    mbcm.update_covs(FC)
    p = mbcm.train_predictor(test_cov=test_cov)

    X_test = np.array(X_test[:ntest], dtype=np.float).copy()
    y_test = y_test[:ntest]

    test_preds = [p(xt.reshape((1, -1)), test_noise_var=test_nv) for xt in X_test]
    test_means, test_vars = zip(*test_preds) 
    test_means = np.asarray(test_means, dtype=float).flatten()
    test_vars = np.asarray(test_vars, dtype=float).flatten()



    n_SX = SX.shape[0]
    outfile = os.path.join(d, "predict_%d_bcm.txt" % n_SX)    
    print outfile
    write_results(outfile, y_test, test_means, test_vars)

    test_preds_local = [p(xt.reshape((1, -1)), test_noise_var=test_nv, local=True) for xt in X_test]
    test_means_local, test_vars_local = zip(*test_preds_local) 
    test_means_local = np.asarray(test_means_local, dtype=float).flatten()
    test_vars_local = np.asarray(test_vars_local, dtype=float).flatten()

    outfile = os.path.join(d, "predict_%d_local.txt" % n_SX)    
    print outfile
    write_results(outfile, y_test, test_means_local, test_vars_local)

def write_results(outfile, y_test, test_means, test_vars):

    y_test = y_test.flatten()

    def gaussian_lp(mean, v, y):
        r = y-mean
        return -.5*r**2/v - .5*np.log(2*np.pi*v)

    r = y_test - test_means
    test_lps = [gaussian_lp(ri, v, 0) for (ri, v) in zip(r, test_vars)]
    meanTest = np.mean(y_test)
    varTest = np.var(y_test)
    stdTest = np.std(y_test)

    baseline_lps = np.array([gaussian_lp(meanTest, stdTest**2, y) for y in y_test])

    mse = np.mean(r **2)
    smse = mse/(varTest+meanTest**2)
    mean_ad = np.mean(np.abs(r))
    median_ad = np.median(np.abs(r))


    msll = np.mean(test_lps - baseline_lps)
    with open(outfile, "w") as f:
        f.write("msll %f\n" % msll)
        f.write("smse: %f\n" % smse)

        f.write("\n")
        f.write("mse: %f\n" % mse)
        f.write("mean_ad: %f\n" % mean_ad)
        f.write("median_ad: %f\n" % median_ad)
        f.write("model lp %f\n" % np.sum(test_lps))
    print "saved results to", outfile

def main():

    parser = argparse.ArgumentParser(description='bcmopt')
    parser.add_argument('--npts', dest='npts', default=-1, type=int)
    parser.add_argument('--block_pts', dest='block_pts', default=300, type=int)
    parser.add_argument('--threshold', dest='threshold', default=0.0, type=float)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int)
    parser.add_argument('--analyze', dest='analyze', default=False, action="store_true")
    parser.add_argument('--predict_from_full', dest='predict_from_full', default=False, action="store_true")
    parser.add_argument('--ntest', dest='ntest', default=-1, type=int)
    args = parser.parse_args()
    if args.analyze:
        eval_predict(args.seed, args.block_pts, args.threshold, args.npts, predict_from_fullX = args.predict_from_full, ntest=args.ntest)
    else:
        run_sarcos(args.seed, args.block_pts, args.threshold, args.npts, args.maxsec)

if __name__ == "__main__":
    main()
