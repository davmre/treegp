import numpy as np
import scipy.optimize
import sys
import copy
import argh
from collections import defaultdict
import cPickle as pickle

from sparsegp.distributions import LogUniform, LogNormal, InvGamma
from sparsegp.gp import GP, GPCov, optimize_gp_hyperparams

from datasets import *

def bfgs_bump(nllgrad, x0, bounds, options, max_rounds=5, **kwargs):

    f1 = 1
    f2 = 0
    rounds = 0
    while f2 <  f1 - .0001 and rounds < max_rounds:
        rounds += 1
        result = scipy.optimize.minimize(fun=nllgrad, x0=x0,
                                         method="L-BFGS-B", jac=True,
                                         options=options, bounds=bounds, **kwargs)

        grad_options = dict()
        grad_options['maxiter'] = 3
        result = scipy.optimize.minimize(fun=nllgrad, x0=result.x,
                                         method="CG", jac=True,
                                         options=grad_options)

        f2 = result.fun
        x0 = result.x

    return result, rounds

def train_hyperparams(X,
                      y,
                      cov_main,
                      cov_fic,
                      noise_var,
                      noise_prior,
                      optimize_xu=False,
                      sparse_invert=False,
                      save_progress = None):

    nllgrad, x0, bounds, build_gp, covs_from_vector = optimize_gp_hyperparams(noise_var=noise_var, cov_main=cov_main, cov_fic=cov_fic, X=X, y=y, noise_prior=noise_prior, optimize_Xu=optimize_xu, sparse_invert=sparse_invert, build_tree=False)

    def nllgrad_checkpoint(v):
        noise_var, cov_main, cov_fic = covs_from_vector(v)
        save_progress(noise_var, cov_main, cov_fic)
        return nllgrad(v)
    f_obj = nllgrad if not save_progress else nllgrad_checkpoint

    result, rounds = bfgs_bump(nllgrad=f_obj, x0=x0,
                       options={'disp': True}, bounds=bounds)

    print "optimized in", rounds, "rounds"
    noise_var, cov_main, cov_fic = covs_from_vector(result.x)
    return noise_var, cov_main, cov_fic

def choose_best_hparams(covs, X, y, noise_prior, sparse_invert=False):

    noise_var, cov_main, cov_fic = covs[0]

    nllgrad, x0, bounds, build_gp, covs_from_vector = optimize_gp_hyperparams(noise_var=noise_var, cov_main=cov_main, cov_fic=cov_fic, X=X, y=y, noise_prior=noise_prior, sparse_invert=sparse_invert, build_tree=False)

    best_ll = np.float('-inf')
    for (noise_var, cov_main, cov_fic) in covs:

        gp = GP(compute_ll=True, noise_var=noise_var,
                cov_main=cov_main, cov_fic=cov_fic, X=X, y=y, sparse_invert=False, build_tree=False)
        ll = gp.ll
        del gp

        ll += noise_prior.log_p(noise_var) + \
              ( cov_main.prior_logp() if cov_main is not None else 0 ) + \
              ( cov_fic.prior_logp() if cov_fic is not None else 0 )

        print "params", noise_var, cov_main, cov_fic, "got likelihood", ll

        if ll > best_ll:
            best_ll = ll
            best_noise_var = noise_var
            best_cov_main = cov_main
            best_cov_fic = cov_fic

    return best_noise_var, best_cov_main, best_cov_fic



def subsample_data(X, y, n):
    p = np.random.permutation(X.shape[0])
    rows = p[:n]
    return X[rows,:], y[rows]


def train_csfic(dataset_name, dfn_params_fic, dfn_params_cs, dfn_str="euclidean", n_train_hyper=1500, n_fic=20, optimize_xu=False, random_restarts=1, dfn_fic_priors=None, dfn_cs_priors=None):
    X_full, y_full = training_data(dataset_name)

    initial_xu, _ = subsample_data(X_full, y_full, n_fic)

    ln = LogNormal(0.0, 2.0)

    if not dfn_cs_priors:
        l1 = LogNormal(0.0, 2.0)
        dfn_cs_priors = [l1,] * len(dfn_params_cs)

    if not dfn_fic_priors:
        l2 = LogNormal(2.0, 3.0)
        dfn_fic_priors = [l2,] * len(dfn_params_fic)

    cov_main = GPCov(wfn_str="compact2", dfn_str=dfn_str,
                      wfn_params=[3.0,], dfn_params=dfn_params_cs,
                      wfn_priors=[ln],
                      dfn_priors=dfn_cs_priors)

    cov_fic = GPCov(wfn_params=[.5,], dfn_params=dfn_params_fic,
                     wfn_str="se", dfn_str=dfn_str, Xu=initial_xu,
                     wfn_priors=[ln],
                     dfn_priors=dfn_fic_priors)
    noise_var = 0.01

    covs = []
    for i in range(random_restarts):

        def save_progress(noise_var, cov_main, cov_fic):
            save_hparams(dataset_name, "csfic%d" % n_fic, cov_main, cov_fic, noise_var, tag="%d_round%d" % (n_train_hyper,i) )


        X, y = subsample_data(X_full, y_full, n_train_hyper)

        noise_var, cov_main, cov_fic = train_hyperparams(X, y, cov_main, cov_fic, noise_var, ln, optimize_xu=optimize_xu, sparse_invert=True, save_progress=save_progress)
        covs.append((noise_var, cov_main, cov_fic))


    X_eval, y_eval = subsample_data(X_full, y_full, n_train_hyper)
    noise_var_best, cov_main_best, cov_fic_best = choose_best_hparams(covs, X_eval, y_eval, ln)

    optim_tag = "_xu" if optimize_xu else ""
    save_hparams(dataset_name, "csfic%d" % n_fic, cov_main_best, cov_fic_best, noise_var_best, tag="%d%s" % (n_train_hyper,optim_tag) )

def train_standard(dataset_name, dfn_params, wfn_str="se", dfn_str="euclidean", n_train_hyper=1500, random_restarts=3, dfn_priors=[], sparse_invert=False):


    X_full, y_full = training_data(dataset_name)

    ln = LogNormal(0.0, 2.0)

    if len(dfn_priors) == 0:
        l1 = LogNormal(0.0, 2.0)
        [l1,] * len(dfn_params)


    cov_main = GPCov(wfn_str=wfn_str, dfn_str=dfn_str,
                      wfn_params=[1.0,], dfn_params=dfn_params,
                      wfn_priors=[ln],
                      dfn_priors=dfn_priors)
    noise_var = 0.01

    covs = []
    for i in range(random_restarts):

        def save_progress(noise_var, cov_main, cov_fic):
            save_hparams(dataset_name, wfn_str, cov_main, cov_fic, noise_var, tag="%d_round%d" % (n_train_hyper,i) )


        X, y = subsample_data(X_full, y_full, n_train_hyper)


        noise_var, cov_main, _,  = train_hyperparams(X, y, cov_main, None, noise_var, ln, sparse_invert=sparse_invert, save_progress=save_progress)
        covs.append((noise_var, cov_main, None))

    X_eval, y_eval = subsample_data(X_full, y_full, n_train_hyper)
    noise_var_best, cov_main_best, _ = choose_best_hparams(covs, X_eval, y_eval, ln)
    save_hparams(dataset_name, wfn_str, cov_main_best, None, noise_var_best, tag="%d" % (n_train_hyper,) )


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


initial_cs_params = {
    "snow": [1.5, .1, .1, 1.5],
    "precip_all": [.2, .2, 2],
    "tco": [25.0, 1.0],
    "housing_age": [2.,2.],
    "housing_val": [2.,2.],
    "housing_inc": [2.,2.],
    "seismic_fitz": [10.0,300.0],
    "seismic_as12": [10.0,300.0],
    "seismic_tt_FITZ": [10.0,300.0],
    "seismic_tt_ASAR": [10.0,300.0],
    "sarcos": [80,] * 21,
}

initial_se_params = {
    "snow": [20., 5., 5., 10.],
    "precip_all": [15.,15.,40.],
    "tco": [187., 14.],
    "housing_age": [8., 8.],
    "housing_val": [8., 8.],
    "housing_inc": [8., 8.],
    "sarcos": [300,] * 21,
    "seismic_fitz": [300.0,300.0],
    "seismic_as12": [300.0,300.0],
    "seismic_tt_FITZ": [300.0,300.0],
    "seismic_tt_ASAR": [300.0,300.0],
}

dfn_cs_priors = defaultdict(list)
#dfn_cs_priors['seismic_fitz'] = [LogNormal(np.log(20), 1.0), LogNormal(np.log(300), 1.0)]
#dfn_cs_priors['seismic_as12'] = [LogNormal(np.log(20), 1.0), LogNormal(np.log(300), 1.0)]
dfn_cs_priors['seismic_as12'] = [InvGamma(10.0, 200.0), InvGamma(10.0, 400.0)]
dfn_cs_priors['seismic_fitz'] = [InvGamma(10.0, 200.0), InvGamma(10.0, 400.0)]
dfn_cs_priors['seismic_tt_ASAR'] = [InvGamma(10.0, 200.0), InvGamma(10.0, 400.0)]
dfn_cs_priors['seismic_tt_FITZ'] = [InvGamma(10.0, 200.0), InvGamma(10.0, 400.0)]

dfn_se_priors = defaultdict(list)
dfn_se_priors['seismic_fitz'] = [LogNormal(np.log(500.0), 1.0), LogNormal(np.log(500.0), 1.0)]
dfn_se_priors['seismic_as12'] = [LogNormal(np.log(500.0), 1.0), LogNormal(np.log(500.0), 1.0)]


def train_hparams(dataset, fic=None, se=False, optimize_xu=False, n_hyper=2500, random_restarts=1):

    dfn_str="euclidean"
    if dataset.startswith("seismic"):
        dfn_str="lld"

    sparse_invert_se = False
    if (dataset == "snow" or dataset == "tco"):
	sparse_invert_se = True

    if fic is None:
        if se:
            train_standard(dataset, initial_se_params[dataset], dfn_str=dfn_str, n_train_hyper=n_hyper, random_restarts=random_restarts, dfn_priors=dfn_se_priors[dataset], wfn_str="se", sparse_invert=sparse_invert_se)
        else:
            train_standard(dataset, initial_cs_params[dataset], dfn_str=dfn_str, n_train_hyper=n_hyper, random_restarts=random_restarts, dfn_priors=dfn_cs_priors[dataset], wfn_str= "compact2", sparse_invert=True)
    else:
        assert(not se)
        train_csfic(dataset, initial_se_params[dataset], initial_cs_params[dataset], dfn_str=dfn_str, n_train_hyper=n_hyper, n_fic=int(fic), random_restarts=random_restarts, dfn_fic_priors=dfn_se_priors[dataset], dfn_cs_priors=dfn_cs_priors[dataset], optimize_xu=optimize_xu)

if __name__ == '__main__':

    argh.dispatch_command(train_hparams)
