import numpy as np
import scipy.optimize
import pandas as pd

import cPickle as pickle

from sparsegp.distributions import LogUniform, LogNormal
from sparsegp.gp import GP, GPCov, optimize_gp_hyperparams

from sparsegp.experiments.datasets import *

def train_hyperparams(X,
                      y,
                      cov_main,
                      cov_fic):

    noise_var_guess = (np.std(y) / 10.0)**2

    nllgrad, x0, bounds, build_gp, covs_from_vector = optimize_gp_hyperparams(noise_var=noise_var_guess, cov_main=cov_main, cov_fic=cov_fic, X=X, y=y, noise_prior=LogUniform(), optimize_Xu=False)

    result = scipy.optimize.minimize(fun=nllgrad, x0=x0,
                                     method="L-BFGS-B", jac=True,
                                     options={'disp': True}, bounds=bounds)


def random_rows(A, n):
    p = np.random.permutation(A.shape[0])
    rows = p[:n]
    return np.copy(A[rows,:]), rows

X, y = training_data('snow')

X, rows = random_rows(X, 500)
y = y[rows]

initial_xu, xu_rows = random_rows(X, 20)
cov_main = GPCov(wfn_str="se", dfn_str="lld",
                 wfn_params=[10.0,], dfn_params=[1000.0, 1000.0],
                 wfn_priors=[LogNormal(1, 2.0)],
                 dfn_priors=[LogNormal(3.0, 2.0), LogNormal(4.0, 4.0)])

cov_fic = GPCov(wfn_params=[1.0,], dfn_params=[100.0, 100.0],
                wfn_str="se", dfn_str="lld", Xu=initial_xu,
                wfn_priors=[LogNormal(0, 2.0)],
                dfn_priors=[LogNormal(5.0, 2.0), LogNormal(4.0, 4.0)])
noise_var=1.0
#train_hyperparams(X, y, cov_main, cov_fic)

save_hparams("snow", "csfic90", cov_main, cov_fic, noise_var)


#if __name__ == '__main__':
#    argh.dispatch_command(train_hyperparams)
