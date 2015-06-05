from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic
from treegp.bcm.local_regression import BCM
from treegp.bcm.bcmopt import cluster_rpc_kernelized, OutOfTimeError

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


from sigvisa.utils.geog import dist_km

COL_IDX, COL_EVID, COL_LON, COL_LAT, COL_SMAJ, COL_SMIN, COL_STRIKE, COL_DEPTH, COL_DEPTHERR = np.arange(9)

def load_seismic_locations():
    fname = "/home/dmoore/python/sigvisa/scraped_partial.txt"
    return np.loadtxt(fname, delimiter=",")

def select_subset(full_data, npts, center=None):
    if center is None:
        center = (82.2904, 46.7937)

    n = full_data.shape[0]
    distances = [dist_km(center, (full_data[i, COL_LON], full_data[i, COL_LAT])) for i in range(n)]
    perm = np.array(sorted(np.arange(n), key = lambda i : distances[i]))
    selected = perm[:npts]
    return full_data[selected]

def load_kernel(fname, sd):
    fullK = np.memmap(fname, dtype='float32', mode='r', shape=(20473, 20473))

    n = sd.shape[0]
    myK = np.zeros((n,n), np.float32)
    for i, sd1 in enumerate(sd):
        idx1 = sd1[COL_IDX]
        myK[i,i] = 1.0
        for j, sd2 in enumerate(sd[:i]):
            idx2 = sd2[COL_IDX]
            min_idx = min(idx1, idx2)
            max_idx = max(idx1, idx2)
            k = fullK[max_idx, min_idx]
            if not np.isfinite(k):
                k = 0.0
            myK[i,j] = k
            myK[j, i] = k
    return myK

def cov_prior(c):
    means = np.array((-5.5, -5.5, 2, 2))
    std = 1.0

    r = (c-means)/std
    ll = -.5*np.sum( r**2)- .5 *len(c) * np.log(2*np.pi*std**2)
    lderiv = -(c-means)/(std**2)
    return ll, lderiv

def do_optimization(d, mbcm, X0, C0, cov_prior, x_prior, maxsec=3600, parallel=False):
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

    cfname = os.path.join(d, "covs.txt")
    covf = open(cfname, 'w')

    def lgpllgrad(x):

        xx = x[:len(x0)]
        xc = x[len(x0):]

        if gradX:
            XX = xx.reshape(X0.shape)
            mbcm.update_X(XX)
            np.save(os.path.join(d, "step_%05d_X.npy" % sstep[0]), XX)
        if gradC:
            FC = np.exp(xc.reshape(C0.shape))
            print FC
            mbcm.update_covs(FC)
            np.save(os.path.join(d, "step_%05d_cov.npy" % sstep[0]), FC)

        ll, gX, gC = mbcm.llgrad(local=True, grad_X=gradX, grad_cov=gradC,
                                       parallel=parallel)
        print "grad", gC

        if gradX:
            prior_ll, prior_grad = x_prior(xx)
            ll += prior_ll
            gX = gX.flatten() + prior_grad
        if gradC:
            prior_ll, prior_grad = cov_prior(xc)
            ll += prior_ll
            gC = (gC * FC).flatten() + prior_grad

        grad = np.concatenate([gX.flatten(), gC.flatten()])

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, ll))
        f_log.flush()

        if gradC:
            covf.write("%d %s\n" % (sstep[0], FC))
            covf.flush()

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            raise OutOfTimeError

        return -ll, -grad

    try:
        bounds = [(None, None),]*len(x0) + [(-12, 8)]*len(c0)
        r = scipy.optimize.minimize(lgpllgrad, full0, jac=True, method="l-bfgs-b", bounds=bounds)
        rx = r.x
    except OutOfTimeError:
        print "terminated optimization for time"

    t1 = time.time()
    f_log.write("optimization finished after %.fs\n" % (time.time()-t0))
    f_log.close()

    covf.close()

    with open(os.path.join(d, "finished"), 'w') as f:
        f.write("")


def build_prior(sd):

    def azimuth_vector(azi_deg):
        angle_deg = 90 - azi_deg
        angle_rad = angle_deg * np.pi/180.0
        return np.array((np.cos(angle_rad), np.sin(angle_rad)))

    def normal_vec(v1):
        return np.array((-v1[1], v1[0]))

    def cov_matrix_km(azi_deg, var1, var2):
        v1 = azimuth_vector(azi_deg).reshape((-1, 1))
        v2 = normal_vec(v1).reshape((-1, 1))
        V = np.hstack((v1, v2))
        D = np.diag((var1, var2))
        return np.dot(V, np.dot(D, V.T))

    def conf_to_var(x, confidence=0.9):
        one_sided = 1.0 - (1.0-confidence)/2
        z = scipy.stats.norm.ppf(one_sided)
        stddev = x/z
        return stddev**2

    def linear_interpolate(x, keys, values):
        idx = np.searchsorted(keys, x, side="right")-1
        k1, k2 = keys[idx], keys[idx+1]
        v1, v2 = values[idx], values[idx+1]
        slope = (v2-v1)/(k2-k1)
        return v1 + (x-k1)*slope

    def degree_in_km(lat):
        # source: http://en.wikipedia.org/wiki/Latitude#Length_of_a_degree_of_latitude
        refs = np.array([0, 15, 30, 45, 60, 75, 90])
        lat_table = [110.574, 110.649, 110.852, 111.132, 111.412, 111.618, 111.694]
        lon_table = [111.320, 107.550, 96.486, 78.847, 55.800, 28.902, 0.000]
        alat = np.abs(lat)
        return linear_interpolate(alat, refs, lon_table), linear_interpolate(alat, refs, lat_table)

    def cov_km_to_deg(lat, C):
        klon, klat = degree_in_km(lat)
        L = np.diag((1.0/klon, 1.0/klat))
        return np.dot(L, np.dot(C, L.T))

    def uncertainty_to_cov(smaj, smin, strike, lat):
        #x1 = ellipse.max_horizontal_uncertainty
        #x2 = ellipse.min_horizontal_uncertainty
        #azi_deg = ellipse.azimuth_max_horizontal_uncertainty
        #C = cov_matrix_km(azi_deg, conf_to_var(x1), conf_to_var(x2))
        if smaj < 0.5:
            smaj = 0.5
        if smin < 0.5:
            smin = 0.5
        C = cov_matrix_km(strike, conf_to_var(smaj), conf_to_var(smin))
        CC = cov_km_to_deg(lat, C)
        return CC

    means = []
    covs = []
    for row in sd:
        means.append((row[COL_LON], row[COL_LAT], row[COL_DEPTH]))
        cov = np.zeros((3,3))
        cov[:2, :2] = uncertainty_to_cov(row[COL_SMAJ], row[COL_SMIN], row[COL_STRIKE], row[COL_LAT])

        deptherr = row[COL_DEPTHERR]
        if deptherr < 1.0:
            deptherr = 1.0
        cov[2,2] = conf_to_var(deptherr)
        covs.append(cov)
    means = np.array(means)

    precs = [np.linalg.inv(cov) for cov in covs]
    logdets = [np.linalg.slogdet(cov)[1] for cov in covs]

    def event_prior_llgrad(x):
        X = x.reshape(means.shape)
        R = X - means
        grad = np.zeros(R.shape)
        ll = 0
        for i, row in enumerate(R):
            alpha = np.dot(precs[i], row)
            ll -= .5 * np.dot(row, alpha)
            ll -= .5 * logdets[i]
            ll -= .5 * np.log(2*np.pi)
            grad[i, :] = -alpha
        return ll, grad.flatten()

    return means.copy(), event_prior_llgrad

def seismic_exp_dir(args):
    npts, block_size, thresh, init_cov, task = args.npts, args.rpc_blocksize, args.threshold, args.init_cov, args.task
    import hashlib
    base_dir = "seismic_experiments"
    run_name = "%d_%d_%.4f_%s_%s" % (npts, block_size, thresh, "default" if init_cov=="" else "_%s" % hashlib.md5(init_cov).hexdigest()[:8], task)
    d =  os.path.join(base_dir, run_name)
    mkdir_p(d)
    return d

def analyze_run_result(args, mbcm):
    d = seismic_exp_dir(args)
    seed = args.seed
    block_size = args.rpc_blocksize
    threshold = args.threshold
    init_cov = args.init_cov
    npts = args.npts
    task = args.task

    steps, times, lls = load_log(d)
    best_step = np.argmax(lls)
    fname_X = os.path.join(d, "step_%05d_X.npy" % best_step)
    X = np.load(fname_X)

    gpll, _, _ = mbcm.gaussian_llgrad_kernel(X, mbcm.YY)

    print "likelihood under true GP", gpll

def main():

    parser = argparse.ArgumentParser(description='seismic')

    parser.add_argument('--npts', dest='npts', default=-1, type=int)

    parser.add_argument('--threshold', dest='threshold', default=0.0, type=float)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int)
    parser.add_argument('--analyze', dest='analyze', default=False, action="store_true")
    parser.add_argument('--rpc_blocksize', dest='rpc_blocksize', default=300, type=int)
    parser.add_argument('--init_cov', dest='init_cov', default="", type=str)
    parser.add_argument('--task', dest='task', default="x", type=str)

    args = parser.parse_args()
    d = seismic_exp_dir(args)
    seed = args.seed
    block_size = args.rpc_blocksize
    threshold = args.threshold
    init_cov = args.init_cov
    npts = args.npts
    task = args.task
    analyze = args.analyze

    fd = load_seismic_locations()
    sd = select_subset(fd, npts)
    YY = load_kernel("xcmax", sd)

    XX = sd[:, [COL_LON, COL_LAT]]
    np.random.seed(seed)
    if block_size > 0:
        SXX, SYY, perm, block_boundaries = cluster_rpc_kernelized(XX, YY, block_size)
        sdp = sd[perm]
    else:
        sdp = sd
        block_boundaries = [(0, npts)]
        perm = np.arange(npts)
        SYY = YY

    SX, x_prior = build_prior(sdp)


    np.save(os.path.join(d, "SX.npy"), SX)
    np.save(os.path.join(d, "perm.npy"), perm)
    np.save(os.path.join(d, "blocks.npy"), np.array(block_boundaries))
    print "saving to", d

    if init_cov == "":
        C0 = np.exp(np.array((-7.5, -5.5, 2.0, 2.0)).reshape((1, -1)))
    else:
        C0 = np.load(init_cov)

    nv = C0[0,0]
    cov = GPCov(wfn_params=[C0[0,1]], dfn_params=C0[0, 2:], dfn_str="lld", wfn_str="matern32")
    mbcm =  MultiSharedBCM(SX, SYY, block_boundaries,
                           cov, nv, dy=200,
                           neighbor_threshold=threshold,
                           nonstationary=False,
                           kernelized=True)

    if task=="x":
        C0 = None
        X0 = SX.copy()
    elif task=="cov":
        X0 = None
    elif task=="xcov":
        X0 = SX.copy()

    if not analyze:
        do_optimization(d, mbcm, X0, C0, cov_prior, x_prior, maxsec=3600, parallel=False)

    import pdb; pdb.set_trace()
    analyze_run_result(args, mbcm)


if __name__ == "__main__":
    main()
