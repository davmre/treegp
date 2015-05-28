from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic, sample_synthetic_bcm_new
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

EXP_DIR = os.path.join(os.environ["HOME"], "bcmopt_experiments")
mkdir_p(EXP_DIR)

MAXSEC=3600

class SampledData(object):

    def __init__(self, old_sdata=None,
                 noise_var=0.01, n=30, ntrain=20, lscale=0.5,
                 obs_std=0.05, yd=10, centers=None, seed=1, samplebcm=False):
        self.noise_var=noise_var
        self.n = n
        self.ntrain = ntrain
        self.lscale=lscale

        if old_sdata is not None:
            self.cov = old_sdata.cov
            self.Xtest, self.Ytest = old_sdata.Xtest, old_sdata.Ytest
            X, Y = old_sdata.SX, old_sdata.SY
        else:
            if samplebcm:
                b = Blocker(centers)
                Xfull, Yfull, cov = sample_synthetic_bcm_new(n=n, noise_var=noise_var, yd=yd, lscale=lscale, seed=seed, blocker = b)
            else:
                Xfull, Yfull, cov = sample_synthetic(n=n, noise_var=noise_var, yd=yd, lscale=lscale, seed=seed)
            self.cov = cov
            X, Y = Xfull[:ntrain,:], Yfull[:ntrain,:]
            self.Xtest, self.Ytest = Xfull[ntrain:,:], Yfull[ntrain:,:]

        if centers is not None:
            b = Blocker(centers)
            self.SX, self.SY, self.perm, self.block_boundaries = b.sort_by_block(X, Y)
            self.centers = centers
        else:
            self.SX, self.SY = X, Y
            self.block_boundaries = [(0, X.shape[0])]
            self.centers = [np.array((0.0, 0.0))]


        self.obs_std = obs_std
        self.X_obs = self.SX + np.random.randn(*X.shape)*obs_std

    def build_mbcm(self, locality=1e-4):
        mbcm = MultiSharedBCM(self.SX, Y=self.SY, block_boundaries=self.block_boundaries, 
                              cov=self.cov, noise_var=self.noise_var, 
                              kernelized=False, neighbor_threshold=locality)
        return mbcm

    def mean_abs_err(self, x):
        return np.mean(np.abs(x - self.SX.flatten()))

    def prediction_error_gp(self, x):
        XX = x.reshape(self.X_obs.shape)
        ll = 0
        for y, yt in zip(self.SY.T, self.Ytest.T):
            gp = GP(X=XX, y=y, cov_main=self.cov, noise_var=self.noise_var,
                    sort_events=False, sparse_invert=False)
            pred_means = gp.predict(self.Xtest)
            pred_cov = gp.covariance(self.Xtest, include_obs=True)
            lly = scipy.stats.multivariate_normal(pred_means, pred_cov).logpdf(yt)
            ll += lly
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


llgrad_bcm = lambda mbcm : mbcm.llgrad(local=True, grad_X=True, parallel=False)
llgrad_lgp = lambda mbcm : mbcm.llgrad_blocked(grad_X=True)
llgrad_gp = lambda mbcm : mbcm.llgrad(grad_X=True)

def do_optimization(llgrad, mbcm, run_name, X0, sdata, method, maxiter=200):

    x0 = X0.flatten()

    sstep = [0,]
    d = os.path.join(EXP_DIR, run_name)
    mkdir_p(d)

    with open(os.path.join(d, "data.pkl"), 'wb') as f:
        pickle.dump(sdata, f)

    f_log = open(os.path.join(d, "log.txt"), 'w')
    t0 = time.time()

    def lgpllgrad(xx):
        XX = xx.reshape(X0.shape)
        mbcm.update_X(XX)
        #ll, grad = mbcm.llgrad(local=True, parallel=True, grad_X=True)
        ll, grad = llgrad(mbcm)
        grad = grad.flatten()

        prior_ll, prior_grad = sdata.x_prior(xx)
        ll += prior_ll
        grad += prior_grad

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, ll))
        f_log.flush()

        np.save(os.path.join(d, "step_%05d.npy" % sstep[0]), XX)
        sstep[0] += 1

        if time.time()-t0 > MAXSEC:
            raise ValueError

        return -ll, -grad

    bounds = [(0.0, 1.0),]*len(x0)
    try:
        r = scipy.optimize.minimize(lgpllgrad, x0, jac=True, method=method, bounds=bounds, options={'maxiter': maxiter})
        rx = r.x
    except ValueError:
        print "terminated optimization for time"
        r = "optimization terminated after %.1fs" % (time.time()-t0)
        rx = np.load(os.path.join(d, "step_%05d.npy" % (sstep[0]-1))).flatten()
    t1 = time.time()

    np.save(os.path.join(d, "step_final"), rx.reshape(X0.shape))

    f_log.close()

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


def run_multi(sdata,run_name,  llgrad, nrestarts):
    mbcm = data.build_mbcm()


def do_run(run_name, lscale, n, ntrain, nblocks, yd, old_sdata=None,
           fullgp=False, restarts=1, method=None, 
           samplebcm=False, obs_std=None, locality=1e-4):
    if fullgp:
        centers = None
        llgrad=llgrad_gp
    else:
        #pmax = np.ceil(np.sqrt(np.sqrt(ntrain)))*2+1
        pmax = np.ceil(np.sqrt(nblocks))*2+1
        pts = np.linspace(0, 1, pmax)[1::2]
        centers = [np.array((xx, yy)) for xx in pts for yy in pts]
        print "bcm with %d blocks" % (len(centers))
        llgrad=llgrad_bcm

    if obs_std is None:
        obs_std = lscale/10

    data = SampledData(noise_var=0.01, n=n,
                       ntrain=ntrain, lscale=lscale,
                       obs_std=obs_std, yd=yd,
                       centers=centers,
                       old_sdata=old_sdata,
                       samplebcm=samplebcm)
    mbcm = data.build_mbcm(locality=locality)
    print mbcm.n_blocks

    if restarts==1:
        X0 = data.X_obs
        do_optimization(llgrad, mbcm, "%s_init00" % (run_name), X0, data, method=method)
    else:
        for i in range(restarts):
            X0 = data.random_init()
            do_optimization(llgrad, mbcm, "%s_init%02d" % (run_name, i), X0, data, method=method)

def savesample(run_name, n, ntrain, lscale, yd):
    data = SampledData(noise_var=0.01, n=n,
                       ntrain=ntrain, lscale=lscale,
                       obs_std=0.05, yd=30,
                       centers=centers)

    d = os.path.join(EXP_DIR, run_name)
    with open(os.path.join(d, "data.pkl"), 'rb') as f:
        old_sdata = pickle.dump(data, f)





if __name__ == "__main__":
    ntrain = int(sys.argv[1])
    n = int(sys.argv[2])
    nblocks = int(sys.argv[3])
    lscale = float(sys.argv[4])
    obs_std = float(sys.argv[5])
    locality = float(sys.argv[6])
    yd = int(sys.argv[7])
    method = sys.argv[8]
    inits =int(sys.argv[9])
    fullgp = sys.argv[10].startswith('t')

    try:
        sdata_fname = sys.argv[11]
        with open(sdata_fname, 'rb') as f:
            sdata = pickle.load(f)
    except IndexError:
        sdata = None

    run_name = "%d_%d_%d_%.2f_%.3f_%.5f_%d_%s_%d_%s" % (ntrain, n, nblocks, lscale, obs_std, locality, yd, method, inits, fullgp)
    d = os.path.join(EXP_DIR, run_name + "_init00")
    print os.path.join(d, "data.pkl")
    do_run(run_name=run_name, lscale=lscale, obs_std=obs_std, locality=locality, n=n, ntrain=ntrain, nblocks=nblocks, yd=yd, fullgp=fullgp, restarts=inits, method=method, samplebcm=False, old_sdata=sdata)
