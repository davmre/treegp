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

    X = np.genfromtxt("raw_X.dat")
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
    gpr.cross_validate(nX, y, kernel="se", folds=5, kernel_params_values= [list(widths),], sigma_values=[0.01, 0.03, 0.1, 0.3, 1, 3])
    


if __name__ == "__main__":
    main()
