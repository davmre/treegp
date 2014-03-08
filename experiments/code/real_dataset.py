import os
import sys
import numpy as np
import scipy.io

from synth_dataset import mkdir_p, eval_gp

from sigvisa.models.spatial_regression.SparseGP import SparseGP, sparsegp_nll_ngrad
from sigvisa.models.distributions import Gaussian
from sigvisa.learn.train_param_common import subsample_data
from sigvisa.infer.optimize.optim_utils import minimize, construct_optim_params

from optparse import OptionParser

def test_predict(sdir, sgp=None, testX=None, testy=None):
    sgp = SparseGP(fname=os.path.join(sdir, "trained.gp"), build_tree=False) if sgp is None else sgp
    testX = np.load(os.path.join(sdir, "testX.npy")) if testX is None else testX
    testy = np.load(os.path.join(sdir, "testy.npy")) if testy is None else testy
    pred_y = sgp.predict_naive(testX)
    predtree_y = sgp.predict(testX)
    r = pred_y - testy

    if np.max(np.abs(pred_y - predtree_y)) > .1:
        X1 = testX[0:1,:]
        Kstar = sgp.kernel(sgp.X, X1)
        gp_pred = np.dot(Kstar.T, sgp.alpha_r)
        import pdb; pdb.set_trace()

    test_lps = np.array([sgp.log_p(x=y, cond=np.reshape(x, (1, -1)), covar='naive') for (x,y) in zip(testX, testy)])

    meanTest = np.mean(testy)
    varTest = np.var(testy)
    stdTest = np.std(testy)
    baseline_model = Gaussian(mean=meanTest, std=stdTest)
    baseline_lps = np.array([baseline_model.log_p(x=y) for y in testy])

    mse = np.mean(r **2)
    smse = mse/(varTest+meanTest**2)
    mean_ad = np.mean(np.abs(r))
    median_ad = np.median(np.abs(r))

    msll = np.mean(test_lps - baseline_lps)

    if msll < 2.0:
        import pdb; pdb.set_trace()

    with open(os.path.join(sdir, "accuracy.txt"), "w") as f:
        f.write("msll %f\n" % msll)
        f.write("smse: %f\n" % smse)

        f.write("\n")
        f.write("mse: %f\n" % mse)
        f.write("mean_ad: %f\n" % mean_ad)
        f.write("median_ad: %f\n" % median_ad)
        f.write("model lp %f\n" % np.sum(test_lps))


def learn_hyperparams(sdir, X, y, hyperparams, dfn_str="euclidean", wfn_str="se", k=500, **kwargs):

    sX, sy = subsample_data(X=X, y=y, k=k)
    print "learning hyperparams on", len(sy), "examples"
    kwargs['build_tree'] = False
    llgrad = lambda p : sparsegp_nll_ngrad(X=sX, y=sy, hyperparams=p, dfn_str=dfn_str, wfn_str=wfn_str, **kwargs)

    bounds = [(1e-20,None),] * len(hyperparams)
    optim_params = construct_optim_params("'disp': True, 'normalize': False, 'bfgs_factor': 1e10")
    params, ll = minimize(f=llgrad, x0=hyperparams, optim_params=optim_params, fprime="grad_included", bounds=bounds)
    return params


def train_realdata_gp(sdir, X, y, hyperparams, dfn_str="euclidean", wfn_str="se", **kwargs):

    hyperparams = np.array(hyperparams, copy=True, dtype=float, order="C")

    sgp = SparseGP(X=X, y=y, hyperparams=hyperparams, dfn_str=dfn_str, wfn_str=wfn_str, build_tree=False, sparse_threshold=1e-8, **kwargs)
    sgp.save_trained_model(os.path.join(sdir, "trained.gp"))
    np.save("K.npy", sgp.K)
    return sgp

def main():

    BASE_DIR = os.path.join(os.getenv('SIGVISA_HOME'), "papers/product_tree/")

    # inputs
    parser = OptionParser()

    parser.add_option("-X", dest="X", default=None, type="str", help="csv file with training locations as rows")
    parser.add_option("--label", dest="label", default=None, type="str", help="dirname of this run")
    parser.add_option("-Y", dest="Y", default=None, type="str", help="csv file with training values as rows")
    parser.add_option("-i", dest="i", default=0, type="int", help="column of Y to use")
    parser.add_option("--n_train", dest="n_train", default=None, type="int", help="number of datapoints")
    parser.add_option("--n_test", dest="n_test", default=None, type="int", help="number of datapoints")
    parser.add_option("--hyperparams", dest="hyperparams", default=".5,10.0", type="str")
    parser.add_option("--dfn_str", dest="dfn_str", default="lld", type="str")

    (options, args) = parser.parse_args()

    X = np.loadtxt(options.X)
    Y = np.loadtxt(options.Y)

    n_train, n_test = options.n_train, options.n_test
    i = options.i
    label = options.label

    if n_train is not None and n_test is not None:
        n = n_train + n_test
    else:
        n = X.shape[0]
        n_train = int(n*.7)
        n_test = int(n*.3)

    np.random.seed(0)
    p = np.random.permutation(n)
    train_idx = p[:n_train]
    test_idx = p[n_train:]

    X_train = np.array(X[train_idx,:], copy=True)
    y_train = np.array(Y[train_idx,i], copy=True)

    X_test = np.array(X[test_idx,:], copy=True)
    y_test = np.array(Y[test_idx,i], copy=True)

    hyperparams = [float(h) for h in options.hyperparams.split(',')]

    model_dir = os.path.join(BASE_DIR, '%s_%d_%d_%d' % (label, i, n_train, n_test))
    mkdir_p(model_dir)

    np.save(os.path.join(model_dir, "testX.npy"), X_test)
    np.save(os.path.join(model_dir, "testy.npy"), y_test)
    np.save(os.path.join(model_dir, "hyperparams.npy"), hyperparams)

    if not os.path.exists(os.path.join(model_dir, 'trained.gp')):
        train_realdata_gp(model_dir, X_train, y_train, dfn_str=options.dfn_str, hyperparams=hyperparams)
        print "trained model"

    test_predict(model_dir)
    print "evaluated predictions"

    eval_gp(bdir=model_dir, test_n=100)
    print "timings finished"

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
