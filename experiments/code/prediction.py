import os
import sys
import numpy as np
import scipy.io

from treegp.gp import GP
from treegp.distributions import Gaussian

from datasets import *

from optparse import OptionParser

def test_predict(dataset_name, model_name, sgp=None, n=None, tag=None):

    sgp = trained_gp(dataset_name, model_name, n=n, tag=tag) if sgp is None else None
    X_test, y_test = test_data(dataset_name)
    print "loaded GP, evaluating predictive performance on %d test points..." % len(y_test)


    if len(y_test) > 5000:
        print "WARNING: using only first 5000 of %d test points" % len(y_test)
        X_test = X_test[:5000, :]
        y_test = y_test[:5000]

    pred_y = sgp.predict_naive(X_test)
    predtree_y = sgp.predict(X_test)
    r = pred_y - y_test

    if np.max(np.abs(pred_y - predtree_y)) > .1:
        X1 = X_test[0:1,:]
        Kstar = sgp.kernel(sgp.X, X1)
        gp_pred = np.dot(Kstar.T, sgp.alpha_r)
        import pdb; pdb.set_trace()

    test_lps = np.array([sgp.log_p(x=y, cond=np.reshape(x, (1, -1)), covar='naive') for (x,y) in zip(X_test, y_test)])

    meanTest = np.mean(y_test)
    varTest = np.var(y_test)
    stdTest = np.std(y_test)
    baseline_model = Gaussian(mean=meanTest, std=stdTest)
    baseline_lps = np.array([baseline_model.log_p(x=y) for y in y_test])

    mse = np.mean(r **2)
    smse = mse/(varTest+meanTest**2)
    mean_ad = np.mean(np.abs(r))
    median_ad = np.median(np.abs(r))

    msll = np.mean(test_lps - baseline_lps)

    outfile = predict_results_fname(dataset_name, model_name, tag)
    with open(outfile, "w") as f:
        f.write("msll %f\n" % msll)
        f.write("smse: %f\n" % smse)

        f.write("\n")
        f.write("mse: %f\n" % mse)
        f.write("mean_ad: %f\n" % mean_ad)
        f.write("median_ad: %f\n" % median_ad)
        f.write("model lp %f\n" % np.sum(test_lps))
    print "saved results to", outfile

def trained_gp(dataset_name, model_name, n=None, tag=None, **kwargs):

    fname = gp_fname(dataset_name, model_name, n=n, tag=tag)
    if os.path.exists(fname):
        return GP(fname=fname, **kwargs)

    X_train, y_train = training_data(dataset_name, n=n)
    cov_main, cov_fic, noise_var = load_hparams(dataset_name, model_name, tag=tag)


    if model_name=="se":
        kwargs['sparse_invert'] = False
        kwargs['build_tree'] = False

    "print training GP for", dataset_name, model_name
    sgp = GP(X=X_train, y=y_train,
             noise_var=noise_var,
             cov_main=cov_main,
             cov_fic=cov_fic,
             sparse_threshold=1e-8,
             **kwargs)
    sgp.save_trained_model(fname)
    print "saved trained GP to", fname

    stats_fname = fname[:-3] + "_stats.txt"
    with open(stats_fname, 'w') as f:
        f.write("main hyperparams: %s\n" % cov_main)
        f.write("fic hyperparams: %s\n" % cov_fic)

        if scipy.sparse.issparse(sgp.K):
            nzr, nzc = sgp.K.nonzero()
            f.write("K nonzero: %d/%d = %.3f%%\n" % (len(nzr), sgp.K.shape[0]**2, float(len(nzr))/sgp.K.shape[0]**2))
        if scipy.sparse.issparse(sgp.Kinv):
            nzr, nzc = sgp.Kinv.nonzero()
            f.write("Kinv nonzero: %d/%d = %.3f%%\n" % (len(nzr), sgp.Kinv.shape[0]**2, float(len(nzr))/sgp.Kinv.shape[0]**2))

        for k in sgp.timings:
            f.write("timing %s: %.3f\n" % (k, sgp.timings[k]))
    print "saved stats to", stats_fname

    return sgp

def main():

    # inputs
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    if len(sys.argv) > 3:
        tag = sys.argv[3]
    else:
        tag = None
    test_predict(dataset_name, model_name, tag=tag)

    #print "timings finished"

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
