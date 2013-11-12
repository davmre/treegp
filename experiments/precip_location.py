import numpy as np
import time
import os
from sigvisa.models.spatial_regression.SparseGP_CSFIC import SparseGP_CSFIC

def x_lp(x, missing_vals, models):

    lp = 0
    for (y, model) in zip(missing_vals, models):
        mylp = model.log_p(x=y, cond=x, covar="double_tree")
        lp += mylp
    return lp

def gridsearch(min_x, max_x, ks, missing_vals, models):

    x1s = np.linspace(min_x[0], max_x[0], ks[0])
    x2s = np.linspace(min_x[1], max_x[1], ks[1])
    x3s = np.linspace(min_x[2], max_x[2], ks[2])

    max_lp = -999
    max_lp_x = []
    lp = np.zeros(ks)
    for (i,x1) in enumerate(x1s):
        for (j,x2) in enumerate(x2s):
            for (k,x3) in enumerate(x3s):
                x = np.array(((x1, x2, x3),))
                lp[i,j,k] = x_lp(x, missing_vals, models)
                if (lp[i,j,k] > max_lp):
                    max_lp = lp[i,j,k]
                    max_lp_x = x

    print "max", max_lp, "at", max_lp_x
    return lp, x1s, x2s, x3s, max_lp_x


def main():


    dfn_str = "euclidean"
    wfn_str_cs = "compact2"
    wfn_str_fic = "se"
    dfn_params_cs = np.array((3.4828263, 2.46854, 42.482933))
    dfn_params_fic = np.array((33.5629, 18.9244, 62.944013)) * 1.41
    wfn_params_cs = np.array((0.3250483,))
    wfn_params_fic = np.array((42.086547,))
    noise_var = .041855
    model_str=""

    """
    dfn_str = "euclidean"
    wfn_str_cs = "compact2"
    wfn_str_fic = "se"
    dfn_params_cs = np.array((1.0, .5, 42.482933))
    dfn_params_fic = np.array((33.5629, 18.9244, 62.944013)) * 1.41
    wfn_params_cs = np.array((0.3250483,))
    wfn_params_fic = np.array((42.086547,))
    noise_var = .041855
    model_str="_fake"
    """

    #Xu = np.loadtxt('run/precip/precip_all_csfic_20/Xu.txt', skiprows=1, delimiter=',')
    Xu = np.loadtxt('run/precip/precip_all_csfic_20/Xu.txt', delimiter=',')

    models = []
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    missing_vals = []
    Xorig = np.loadtxt('datasets/precip/precip_all_X.txt', skiprows=1, delimiter=',')

    np.random.seed(0)
    p = np.random.permutation(Xorig.shape[0])
    X = np.array(Xorig[p[:-100], :], copy=True)

    missing_row = p[-100]
    true_missing_x = Xorig[missing_row, :]
    print "true", true_missing_x
    missing_vals = []
    for month in months:

        y = np.loadtxt('datasets/precip/precip_%s_y.txt' % month, skiprows=1, delimiter=',')

        missing_vals.append(y[missing_row])

        y = np.array(y[p[:-100]], copy=True)


        fname = "model_%s%s.gp" % (month, model_str)
        if os.path.exists(fname):
            model = SparseGP_CSFIC(fname=fname, build_tree=True, leaf_bin_size=15)
        else:
            model = SparseGP_CSFIC(X=X, y=y,
                                   dfn_str=dfn_str, dfn_params_fic=dfn_params_fic, dfn_params_cs=dfn_params_cs,
                                   wfn_str_fic = wfn_str_fic, wfn_params_fic=wfn_params_fic,
                                   wfn_str_cs = wfn_str_cs, wfn_params_cs=wfn_params_cs,
                                   build_tree=True, sort_events=True, center_mean=False,
                                   noise_var=noise_var, Xu=Xu, leaf_bin_size=15)
            model.save_trained_model(fname)
        models.append(model)


    t0 = time.time()
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)
    print min_x
    print max_x
    lp, x1s, x2s, x3s, best_x = gridsearch(min_x, max_x, np.array((20, 20, 7)), np.array(missing_vals), models)
    t1 = time.time()
    print "first gridsearch took", t1-t0

    np.savez("gridsearch1.npz", lp=lp, x1s=x1s, x2s=x2s, x3s=x3s)

    min_x = best_x - np.array([2.5, 2.5, 5.0])
    max_x = best_x + np.array([2.5, 2.5, 5.0])
    lp2, x1s, x2s, x3s, best_x = gridsearch(min_x.flatten(), max_x.flatten(), np.array((20, 20, 7)), np.array(missing_vals), models)
    t2 =time.time()
    print best_x
    np.savez("gridsearch2.npz", lp=lp2, x1s=x1s, x2s=x2s, x3s=x3s)
    print  "second gridsearch took", t2-t1
    print "total time", t2-t0

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
