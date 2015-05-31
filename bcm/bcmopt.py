from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic
from treegp.bcm.local_regression import BCM

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from treegp.util import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import time
import os
import sys
import cPickle as pickle

import argparse

EXP_DIR = os.path.join(os.environ["HOME"], "bcmopt_experiments")

def sample_data(n, ntrain, lscale, obs_std, yd, seed, centers):
    sample_basedir = os.path.join(os.environ["HOME"], "bcmopt_experiments", "synthetic_datasets")
    mkdir_p(sample_basedir)
    sample_fname = "%d_%d_%.2f_%.3f_%d_%d.pkl" % (n, ntrain, lscale, obs_std, yd, seed)
    sample_fname_full = os.path.join(sample_basedir, sample_fname)

    try:
        with open(sample_fname_full, 'rb') as f:
            sdata = pickle.load(f)
    except IOError:
        sdata = SampledData(n=n, ntrain=ntrain, lscale=lscale, obs_std=obs_std, seed=seed, centers=None)

        with open(sample_fname_full, 'wb') as f:
            pickle.dump(sdata, f)


    sdata.set_centers(centers)
    return sdata

class OutOfTimeError(Exception):
    pass

class SampledData(object):

    def __init__(self,
                 noise_var=0.01, n=30, ntrain=20, lscale=0.5,
                 obs_std=0.05, yd=10, centers=None, seed=1):
        self.noise_var=noise_var
        self.n = n
        self.ntrain = ntrain
        self.lscale=lscale

        Xfull, Yfull, cov = sample_synthetic(n=n, noise_var=noise_var, yd=yd, lscale=lscale, seed=seed)
        self.cov = cov
        X, Y = Xfull[:ntrain,:], Yfull[:ntrain,:]
        self.Xtest, self.Ytest = Xfull[ntrain:,:], Yfull[ntrain:,:]

        self.SX, self.SY = X, Y
        self.block_boundaries = [(0, X.shape[0])]
        self.centers = [np.array((0.0, 0.0))]

        self.obs_std = obs_std
        np.random.seed(seed)
        self.X_obs = self.SX + np.random.randn(*X.shape)*obs_std

    def set_centers(self, centers):
        b = Blocker(centers)
        self.SX, self.SY, self.perm, self.block_boundaries = b.sort_by_block(self.SX, self.SY)
        self.centers = centers
        self.X_obs = self.X_obs[self.perm]

    def build_mbcm(self, local_dist=1e-4):
        mbcm = MultiSharedBCM(self.SX, Y=self.SY, block_boundaries=self.block_boundaries,
                              cov=self.cov, noise_var=self.noise_var,
                              kernelized=False, neighbor_threshold=local_dist)
        return mbcm

    def mean_abs_err(self, x):
        return np.mean(np.abs(x - self.SX.flatten()))

    def prediction_error_gp(self, x):
        XX = x.reshape(self.X_obs.shape)
        ntest = self.n-self.ntrain
        ll = 0

        gp = GP(X=XX, y=self.SY[:, 0:1], cov_main=self.cov, noise_var=self.noise_var,
                sort_events=False, sparse_invert=False)
        pred_cov = gp.covariance(self.Xtest, include_obs=True)
        logdet = np.linalg.slogdet(pred_cov)[1]
        pred_prec = np.linalg.inv(pred_cov)

        for y, yt in zip(self.SY.T, self.Ytest.T):
            gp.y = y
            gp.alpha_r = gp.factor(y)
            pred_means = gp.predict(self.Xtest)

            lly =  -.5 * np.dot(yt, np.dot(pred_prec, yt))
            lly += -.5 * logdet
            lly += -.5 * ntest * np.log(2*np.pi)

            ll += lly

        return ll

    def prediction_error_bcm(self, x, test_cov, local_dist=1.0):
        ntest = self.n-self.ntrain
        mbcm = self.build_mbcm(self, local_dist=local_dist)

        # TODO: predict with respect to local covs instead of a global test cov
        p = mbcm.train_predictor(test_cov)
        PM, PC = p(predict, test_noise_var=self.noise_var)
        PP = np.linalg.inv(PC)

        ll =  -.5 * np.sum(PP, np.dot(PM, PM.T))
        ll += -.5 * np.linalg.slogdet(PC)[1]
        ll += -.5 * ntest * self.yd * np.log(2*np.pi)
        return ll

    def prediction_error_bcm(self, x):
        XX = x.reshape(self.X_obs.shape)
        ll = 0
        for y, yt in zip(self.SY.T, self.Ytest.T):
            bcm = BCM(block_centers=self.centers, cov_block_params = [(self.noise_var, 1.0, self.lscale, self.lscale)],
                      X=XX, y=y, test_cov=self.cov)
            m, c = bcm.predict_dist(self.Xtest, noise_var=self.noise_var)
            lly = scipy.stats.multivariate_normal(m, c).logpdf(yt)
            ll += lly
        return ll

    def x_prior(self, xx):
        flatobs = self.X_obs.flatten()
        t0 = time.time()

        n = len(xx)
        r = (xx-flatobs)/self.obs_std
        ll = -.5*np.sum( r**2)- .5 *n * np.log(2*np.pi*self.obs_std**2)

        lderiv = np.array([-(xx[i]-flatobs[i])/(self.obs_std**2) for i in range(len(xx))]).flatten()
        t1 = time.time()
        return ll, lderiv

    def random_init(self, jitter_std=None):
        if jitter_std is None:
            jitter_std = self.obs_std
        return self.X_obs + np.random.randn(*self.X_obs.shape)*jitter_std


def do_optimization(d, mbcm, X0, C0, sdata, method, maxsec=3600, parallel=False):

    gradX = (X0 is not None)
    gradC = (C0 is not None)

    if gradX:
        x0 = X0.flatten()
    else:
        x0 = np.array(())

    if gradC:
        c0 = np.log(C0.flatten())
    else:
        c0 = np.array(())
    full0 = np.concatenate([x0, c0])

    sstep = [0,]
    f_log = open(os.path.join(d, "log.txt"), 'w')
    t0 = time.time()

    def lgpllgrad(x):

        xx = x[:len(x0)]
        xc = x[len(x0):]

        if gradX:
            XX = xx.reshape(X0.shape)
            mbcm.update_X(XX)
            np.save(os.path.join(d, "step_%05d_X.npy" % sstep[0]), XX)
        if gradC:
            C = np.exp(xc.reshape(C0.shape))
            mbcm.update_covs(C)
            np.save(os.path.join(d, "step_%05d_cov.npy" % sstep[0]), C)

        ll, gX, gC = mbcm.llgrad(local=True, grad_X=gradX, grad_cov=gradC,
                                       parallel=parallel)

        if gradX:
            prior_ll, prior_grad = sdata.x_prior(xx)
            ll += prior_ll
            gX = gX.flatten() + prior_grad
        if gradC:
            gC = np.array(gradC) * C

        grad = np.concatenate([gX.flatten(), gC.flatten()])

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, ll))
        f_log.flush()

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            raise OutOfTimeError

        return -ll, -grad

    bounds = [(0.0, 1.0),]*len(x0) + [(-10, 5)]*len(c0)
    try:
        r = scipy.optimize.minimize(lgpllgrad, full0, jac=True, method=method, bounds=bounds)
        rx = r.x
    except OutOfTimeError:
        print "terminated optimization for time"

    t1 = time.time()
    f_log.write("optimization finished after %.fs\n" % (time.time()-t0))
    f_log.close()

    with open(os.path.join(d, "finished"), 'w') as f:
        f.write("")

def analyze_run(n, ntrain, lscale, obs_std, yd, seed):
    with open(os.path.join(d, "results.txt"), 'w') as f:
        f.write("optimized in %.2f seconds\n" % (t1-t0))
        f.write("MAD error %.04f to %.04f\n" % (sdata.mean_abs_err(x0), sdata.mean_abs_err(rx)))
        if X0.shape[0] < 1000:
            f.write("GP predictive likelihood %.3f to %.3f (true %.3f)\n" % (sdata.prediction_error_gp(x0),
                                                                             sdata.prediction_error_gp(rx),
                                                                             sdata.prediction_error_gp(sdata.SX.flatten())))
            f.write("BCM predictive likelihood %.3f to %.3f (true %.3f)\n" % (sdata.prediction_error_bcm(x0),
                                                                          sdata.prediction_error_bcm(rx),
                                                                          sdata.prediction_error_bcm(sdata.SX.flatten())))
        f.write("\n\nresult:\n")
        f.write(str(r))
        f.write("\n")


def do_run(run_name, lscale, n, ntrain, nblocks, yd, seed=0,
           fullgp=False, method=None,
           obs_std=None, local_dist=1.0, maxsec=3600):

    pmax = np.ceil(np.sqrt(nblocks))*2+1
    pts = np.linspace(0, 1, pmax)[1::2]
    centers = [np.array((xx, yy)) for xx in pts for yy in pts]
    print "bcm with %d blocks" % (len(centers))

    if obs_std is None:
        obs_std = lscale/10

    data = sample_data(n=n, ntrain=ntrain, lscale=lscale, obs_std=obs_std, yd=yd, seed=seed, centers=centers)
    mbcm = data.build_mbcm(local_dist=local_dist)

    d = os.path.join(EXP_DIR, run_name)
    mkdir_p(d)

    X0 = data.X_obs
    do_optimization(d, mbcm, X0, None, data, method=method, maxsec=maxsec)

def build_run_name(args):
    run_name = "%d_%d_%d_%.2f_%.3f_%.5f_%d_%s" % (args.ntrain, args.n, args.nblocks, args.lscale, args.obs_std, args.local_dist, args.yd, args.method)
    return run_name

def main():

    mkdir_p(EXP_DIR)

    MAXSEC=3600

    parser = argparse.ArgumentParser(description='bcmopt')
    parser.add_argument('--ntrain', dest='ntrain', type=int)
    parser.add_argument('--n', dest='n', type=int)
    parser.add_argument('--nblocks', dest='nblocks', default=1, type=int)
    parser.add_argument('--lscale', dest='lscale', type=float)
    parser.add_argument('--obs_std', dest='obs_std', type=float)
    parser.add_argument('--local_dist', dest='local_dist', default=1.0, type=float)
    parser.add_argument('--method', dest='method', default="l-bfgs-b", type=str)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--yd', dest='yd', default=50, type=int)
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int)

    args = parser.parse_args()

    run_name = build_run_name(args)
    do_run(run_name=run_name, lscale=args.lscale, obs_std=args.obs_std, local_dist=args.local_dist, n=args.n, ntrain=args.ntrain, nblocks=args.nblocks, yd=args.yd, method=args.method, seed=args.seed, maxsec=args.maxsec)

if __name__ == "__main__":
    main()
