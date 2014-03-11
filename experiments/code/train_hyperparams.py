import numpy as np
import scipy.optimize
import pandas as pd

import cPickle as pickle

from sparsegp.distributions import LogUniform, LogNormal
from sparsegp.gp import GP, GPCov, optimize_gp_hyperparams

from datasets import *

def train_hyperparams(X,
                      y,
                      cov_main,
                      cov_fic,
                      noise_var,
                      noise_prior,
                      optimize_xu=False):

    nllgrad, x0, bounds, build_gp, covs_from_vector = optimize_gp_hyperparams(noise_var=noise_var, cov_main=cov_main, cov_fic=cov_fic, X=X, y=y, noise_prior=noise_prior, optimize_Xu=optimize_xu)

    result = scipy.optimize.minimize(fun=nllgrad, x0=x0,
                                     method="L-BFGS-B", jac=True,
                                     options={'disp': True}, bounds=bounds)

    return covs_from_vector(result.x)


def random_rows(A, n):
    p = np.random.permutation(A.shape[0])
    rows = p[:n]
    return np.copy(A[rows,:]), rows


def train_generic(dataset_name, dfn_params_fic, dfn_params_cs, dfn_str="euclidean", n_train_hyper=1500, n_fic=20, optimize_xu=False):
    X, y = training_data(dataset_name)

    X, rows = random_rows(X, n_train_hyper)
    y = y[rows]

    initial_xu, xu_rows = random_rows(X, n_fic)
    ln = LogNormal(0.0, 2.0)
    l1 = LogNormal(0.0, 2.0)
    l2 = LogNormal(2.0, 2.0)
    cov0_main = GPCov(wfn_str="compact2", dfn_str=dfn_str,
                      wfn_params=[1.0,], dfn_params=dfn_params_cs,
                      wfn_priors=[ln],
                      dfn_priors=[l1,] * len(dfn_params_cs))

    cov0_fic = GPCov(wfn_params=[1.0,], dfn_params=dfn_params_fic,
                     wfn_str="se", dfn_str=dfn_str, Xu=initial_xu,
                     wfn_priors=[ln],
                     dfn_priors=[l2,] * len(dfn_params_fic))
    noise_var, cov_main, cov_fic = train_hyperparams(X, y, cov0_main, cov0_fic, 0.5, ln, optimize_xu=optimize_xu)
    optim_tag = "_xu" if optimize_xu else ""
    save_hparams(dataset_name, "csfic%d" % n_fic, cov_main, cov_fic, noise_var, tag="%d%s" % (n_train_hyper,optim_tag) )

def hardcode_snow():

    initial_xu = [[178,120.0021,-38.6924,51.634921],
                  [184,119.377,-38.44,65.079365],
                  [65,122.5277,41.3023,32.539683],
                  [55,119.5356,-38.3975,66.873016],
                  [192,119.238,-37.555,44.444444],
                  [20,122.5277,-41.3023,32.539683],
                  [1,121.321,-39.813,0.79365079],
                  [52,118.937,-37.183,73.015873],
                  [78,120.678,-39.623,21.904762],
                  [130,120.118,-38.678,31.746032],
                  [151,118.562,-37.176,71.428571],
                  [170,118.442,-36.497,82.539683],
                  [85,121.198,-40.77,30.952381],
                  [68,119.662,-38.158,52.380952],
                  [35,120.197,-38.925,55.555556],
                  [144,122.8,-41.008,20.634921],
                  [205,118.562,-37.162,80.952381],
                  [50,120.118,-38.678,31.746032],
                  [13,122.5277,41.3023,32.539683],
                  [116,119.234,-38.077,66.666667]]

    cov_main = GPCov(wfn_str="compact2", dfn_str="euclidean",
                      wfn_params=[16.26,], dfn_params=[36.361, .111, .120, 1.642])

    cov_fic = GPCov(wfn_params=[4.831,], dfn_params=[61.563, 6.56, 75.187, 108.456],
                     wfn_str="se", dfn_str="euclidean", Xu=initial_xu)
    save_hparams("snowm", "csfic20" , cov_main, cov_fic, .1377, tag="matlab" )

if __name__ == '__main__':
    train_generic("snow", [20, 5, 5, 10], [1.5, .1, .1, 1.5], n_train_hyper=5000)

    train_generic("precip_all", [15, 15, 40], [.2, .2, 2], n_train_hyper=5000, n_fic=20)
    train_generic("precip_all", [15, 15, 40], [.2, .2, 2], n_train_hyper=5000, n_fic=90)

    train_generic("tco", [40, 40], [2.0, 2.0], n_train_hyper=5000, n_fic=20)
    train_generic("tco", [40, 40], [2.0, 2.0], n_train_hyper=5000, n_fic=90)

    train_generic("housing_age", [8, 8], [2,2], n_train_hyper=5000, n_fic=20)
    train_generic("housing_age", [8, 8], [2,2], n_train_hyper=5000, n_fic=90)
    train_generic("housing_val", [8, 8], [2,2], n_train_hyper=5000, n_fic=20)
    train_generic("housing_val", [8, 8], [2,2], n_train_hyper=5000, n_fic=90)
    train_generic("housing_inc", [8, 8], [2,2], n_train_hyper=5000, n_fic=20)
    train_generic("housing_inc", [8, 8], [2,2], n_train_hyper=5000, n_fic=90)


#    argh.dispatch_command(train_hyperparams)
