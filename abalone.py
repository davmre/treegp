import numpy as np

import munge
import gp_regression as gpr

def process_data():
    convertfunc = lambda s: float(0.0) if s=='M' else float(1.0) if s=='F' else float(2.0) if s=='I' else float(-1)
    data = np.genfromtxt("abalone.data", delimiter=',', converters={0:convertfunc})
    data = data.view(float).reshape((data.shape[0], -1))
    idata, d = munge.categorical_to_indicators(data, 0)

    X, y, nX, ny = munge.preprocess(idata, target=(idata.shape[1]-1))
    np.savetxt("raw_X.dat", X)
    np.savetxt("raw_y.dat", y)
    np.savetxt("cooked_X.dat", nX)
    np.savetxt("cooked_y.dat", ny)

def main():
#    process_data()

#    X = np.genfromtxt("raw_X.dat")
    y = np.genfromtxt("raw_y.dat")
    nX = np.genfromtxt("cooked_X.dat")
#    ny = np.genfromtxt("cooked_y.dat")

    #distance_quantiles(X, [0.05, 0.1, 0.5, 0.9, 0.95])
    widths = munge.distance_quantiles(nX, [0.01, 0.05, 0.1, 0.5, 0.9, 0.95])

#    gp1 = gpr.GaussianProcess(X = nX[1:10, :], y = y[1:10], sigma = 0.1, kernel="sqexp", kernel_params=(22,))
#    gp1.save_trained_model("model.npz")

#    print "loading"
#    gp2 = gpr.GaussianProcess(fname="model.npz")
#    gp2.cross_validate(kernel_params_values= [list(widths)])
#    print X.shape, nX.shape, y.shape, ny.shape
#    print widths

#    gpr.cross_validate(X, y, kernel="sqexp", folds=5, kernel_params_values= [list(widths),], sigma_values=[0.005, 0.01, 0.03, 0.1, 0.3, 1, 3])
#    gpr.cross_validate(nX, y, kernel="se", folds=5, kernel_params_values= [list(widths),], sigma_values=[0.01, 0.03, 0.1, 0.3, 1, 3])
    """
    X = np.array([
        [ -4.0 ],
        [ -3.0 ],
        [ -1.0 ],
        [  0.0 ],
        [  2.0 ],
        ])
    y = np.array( [
            -2.0,
             0.0,
             1.0,
             2.0,
             -1.0
             ] )

    ip = np.array((-4.38027342683e-07, 1.08346151297, 0.798937185643))
    sp = np.array((.2, 1, 1))
    best_params = gpr.optimize_hyperparams(X, y, "se", sp)
    best_params = ip

    gp = gpr.GaussianProcess(X = X, y = y, kernel="se", kernel_params=best_params)
    print "best params", best_params

    gpr.gp_1D_predict(gp, x_min = -5.0, x_max = 5.0 )"""
    #    print gp.log_likelihood()
#    gp.save_trained_model("testK.npz")
    

#    best_params = gpr.optimize_hyperparams(nX[0:100], y[0:100], "se", np.array((1, 1, 100, 100,100,100,100,100,100,100,100,100)))
#    print best_params
#    kl, tl, bl = gpr.test_kfold(nX, y, 5, "se_iso", best_params, None, gpr.sq_loss, train_loss=True)
#    print kl, tl, bl

    start = np.array([2.07022092e+00,   3.59469523e+03,   1.00322668e+02,   9.40285748e+01,   8.78873675e+01,   7.83023525e+01,   7.37698270e+01,   6.38871011e+01,   6.95504727e+01,   3.42247272e+01,   8.84079900e+01,   3.29584700e+01])

#    p, ll = gpr.optimize_hyperparams(nX, y, "se", np.array((.01, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100)))
    p, ll = gpr.optimize_hyperparams(nX, y, "se", start)
    print "first try gives", p, ll
    p, ll = gpr.optimize_hyperparams(nX, y, "se", np.array((1, 1, 100)))
    print "second try gives", p, ll
    p, ll = gpr.optimize_hyperparams(nX, y, "se", np.array((.1, 1, 10)))
    print "third try gives", p, ll
    p, ll = gpr.optimize_hyperparams(nX, y, "se", np.array((1, 1, 10)))
    print "fourth try gives", p, ll


#    gp.save_trained_model("testK.npz")
    

if __name__ == "__main__":
    main()
    
