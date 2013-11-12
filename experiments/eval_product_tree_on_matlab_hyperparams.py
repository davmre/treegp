import os
import sys
import numpy as np
import scipy.io

from synth_dataset import mkdir_p, eval_gp
from real_dataset import test_predict

from sigvisa.models.spatial_regression.SparseGP import SparseGP, sparsegp_nll_ngrad
from sigvisa.models.spatial_regression.SparseGP_CSFIC import SparseGP_CSFIC
from sigvisa.models.distributions import Gaussian
from sigvisa.learn.train_param_common import subsample_data
from sigvisa.infer.optimize.optim_utils import minimize, construct_optim_params

def sort_events(X, y):
    combined = np.hstack([X, np.reshape(y, (-1, 1))])
    combined_sorted = np.array(sorted(combined, key = lambda x: x[0]), dtype=float)
    X_sorted = np.array(combined_sorted[:, :-1], copy=True, dtype=float)
    y_sorted = combined_sorted[:, -1].flatten()
    return X_sorted, y_sorted

def load_matlab_csficmodel(csficbase, X_train, y_train):


    dfn_str = "euclidean"
    wfn_str_fic = "se"
    wfn_str_cs = "compact2"


    dfn_params_cs = np.loadtxt(os.path.join(csficbase, 'dfn_params_cs.txt'), delimiter=',', ndmin=1)
    dfn_params_fic = np.loadtxt(os.path.join(csficbase, 'dfn_params_fic.txt'), delimiter=',', ndmin=1)
    dfn_params_fic *= np.sqrt(2.0)
    wfn_params_cs = np.loadtxt(os.path.join(csficbase, 'wfn_params_cs.txt'), delimiter=',', ndmin=1)
    wfn_params_fic = np.loadtxt(os.path.join(csficbase, 'wfn_params_fic.txt'), delimiter=',', ndmin=1)
    noise_var_csfic = np.loadtxt(os.path.join(csficbase, 'noise_var.txt'), delimiter=',')
    try:
        Xu = np.loadtxt(os.path.join(csficbase, 'Xu.txt'), delimiter=',')
    except:
        print "didn't find Xu, creating new ones!"
        p = np.random.permutation(X_train.shape[0])
        Xu = np.array(X_train[p[:90], :], copy=True)

    gp_csfic = SparseGP_CSFIC(X=X_train, y=y_train,
                              dfn_str=dfn_str, dfn_params_fic=dfn_params_fic, dfn_params_cs=dfn_params_cs,
                              wfn_str_fic = wfn_str_fic, wfn_params_fic=wfn_params_fic,
                              wfn_str_cs = wfn_str_cs, wfn_params_cs=wfn_params_cs,
                              build_tree=True, sort_events=True, center_mean=False,
                              noise_var=noise_var_csfic, Xu=Xu, leaf_bin_size=15)
    return gp_csfic

def load_matlab_semodel(basename, X_train, y_train):

    dfn_str = "euclidean"
    wfn_str = "se"

    sebase = basename + '_se'
    dfn_params_se = np.loadtxt(os.path.join(sebase, 'dfn_params.txt'), delimiter=',', ndmin=1)
    dfn_params_se *= np.sqrt(2.0)
    wfn_params_se = np.loadtxt(os.path.join(sebase, 'wfn_params.txt'), delimiter=',', ndmin=1)
    noise_var_se = np.loadtxt(os.path.join(sebase, 'noise_var.txt'), delimiter=',')

    hyperparams = np.concatenate([[noise_var_se, ], wfn_params_se, dfn_params_se])

    gp_se = SparseGP(X=X_train, y=y_train,
                     dfn_str=dfn_str,
                     wfn_str = wfn_str, hyperparams=hyperparams,
                     build_tree=True, sort_events=True, center_mean=False, sparse_invert=False)
    print "trained SE model"
    return gp_se


def main(n_max=18000):

    rundir = sys.argv[1]
    task_name = sys.argv[2]



    basedir = os.path.join(os.getenv('SIGVISA_HOME'), 'papers', 'product_tree', 'run', rundir)
    basename = os.path.join(basedir, task_name)
    X_train = np.loadtxt(basename + '_X_train.txt', delimiter=',')
    y_train = np.loadtxt(basename + '_y_train.txt',  delimiter=',')

    n_X = X_train.shape[0]
    if n_X > n_max:
        X_train = np.array(X_train[:n_max,:], copy=True)
        y_train = np.array(y_train[:n_max], copy=True)
        print "using restricted subset of %d training points" % (n_max)
    actual_n = min(n_X, n_max)


    X_test = np.loadtxt(basename + '_X_test.txt',  delimiter=',')
    y_test = np.loadtxt(basename + '_y_test.txt',  delimiter=',')

    """
    #for nu in (90,):
    nu = 90
    csficbase = basename + '_csfic_%d' % nu if nu is not None else basename + '_csfic'
    csficmodel_dir = basename + "_py_csfic_%d" % nu if nu is not None else basename + "_py_csfic"
    print csficmodel_dir
    #if not os.path.exists(csficbase):
    #    continue
    csgp = os.path.join(csficmodel_dir, 'trained_%d.gp' % actual_n)

    mkdir_p(csficmodel_dir)
    if os.path.exists(csgp):
        gp_csfic = SparseGP_CSFIC(fname=csgp, build_tree=True, leaf_bin_size=15)
    else:
        gp_csfic = load_matlab_csficmodel(csficbase, X_train, y_train)
        gp_csfic.save_trained_model(csgp)

    #print "testing predictions"
    #test_predict(csficmodel_dir, sgp=gp_csfic, testX=X_test, testy=y_test)

    #print "testing cutoff rule 0"
    #eval_gp(bdir=csficmodel_dir, testX=X_test, test_n=200, gp=gp_csfic, cutoff_rule=0)
    #print "testing cutoff rule 1"
    #eval_gp(bdir=csficmodel_dir, testX=X_test, test_n=200, gp=gp_csfic, cutoff_rule=1)
    #print "testing cutoff rule 2"
    #eval_gp(bdir=csficmodel_dir, testX=X_test, test_n=200, gp=gp_csfic, cutoff_rule=2)

    print "testing leaf bins 15"
    gp_csfic10 = SparseGP_CSFIC(fname=csgp, build_tree=True, leaf_bin_size=10)
    eval_gp(bdir=csficmodel_dir, testX=X_test, test_n=200, gp=gp_csfic10, cutoff_rule=2, flag="_bin15")

    #print "testing leaf bins 100"
    #gp_csfic50 = SparseGP_CSFIC(fname=csgp, build_tree=True, leaf_bin_size=50)
    #eval_gp(bdir=csficmodel_dir, testX=X_test, test_n=200, gp=gp_csfic50, cutoff_rule=2, flag="_bin50")



    """
    semodel_dir = basename + "_py_se"
    segp = os.path.join(semodel_dir, 'trained.gp')
    mkdir_p(semodel_dir)
    if os.path.exists(segp):
        gp_se = SparseGP(fname=segp, build_tree=True, leaf_bin_size=0, sparse_invert=False)
    else:
        gp_se = load_matlab_semodel(basename, X_train, y_train)
        gp_se.save_trained_model(segp)

    test_predict(semodel_dir, sgp=gp_se, testX=X_test, testy=y_test)
    eval_gp(bdir=semodel_dir, testX=X_test, test_n=200, gp=gp_se)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import sys, traceback, pdb
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
