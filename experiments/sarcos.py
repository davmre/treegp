import os
import sys
import numpy as np
import scipy.io
import time

from synth_dataset import mkdir_p, eval_gp
from real_dataset import train_realdata_gp, test_predict
from sigvisa.models.spatial_regression.SparseGP import SparseGP


def load_sarcos(sdir, train_n=4096):
    sarcos_train = scipy.io.loadmat(os.path.join(sdir, 'sarcos_inv.mat'))['sarcos_inv'].byteswap().newbyteorder()
    sarcos_test = scipy.io.loadmat(os.path.join(sdir, 'sarcos_inv_test.mat'))['sarcos_inv_test'].byteswap().newbyteorder()

    # results from one trial of hyperparam optimization using Krzysztof Chalupka's MATLAB code
    results = scipy.io.loadmat(os.path.join(sdir, "resultsSoD_SARCOS.mat"))
    hyperparams = results["resultsSoD"]["hyps"].item()[0, 0]
    my_hyperparams = np.zeros((len(hyperparams),))
    my_hyperparams[0] = np.exp(hyperparams[-1])**2 # noise variance
    my_hyperparams[1] = np.exp(hyperparams[-2])**2 # signal variance
    my_hyperparams[2:] = np.exp(hyperparams[:-2, 0]) /5.0 # lengthscales
    hyperparams = my_hyperparams

    sarcos_train_X = sarcos_train[:, :21]
    sarcos_train_y = sarcos_train[:, 21]
    sarcos_test_X = sarcos_test[:, :21]
    sarcos_test_y = sarcos_test[:, 21]

    train_X_mean = np.reshape(np.mean(sarcos_train_X, axis=0), (1, -1))
    train_X_std = np.reshape(np.std(sarcos_train_X, axis=0), (1, -1))
    train_y_mean = np.mean(sarcos_train_y)
    train_y_std = np.std(sarcos_train_y)

    sod = results["resultsSoD"]["sod"].item()[0,0].flatten()
    perm = sod - 1
    #perm = np.random.permutation(len(sarcos_train))[0:train_n]
    sarcos_train_X = sarcos_train_X[perm,:]
    sarcos_train_y = sarcos_train_y[perm]

    sarcos_train_X = (sarcos_train_X - train_X_mean)/train_X_std
    sarcos_test_X = (sarcos_test_X - train_X_mean)/train_X_std
    sarcos_train_y = (sarcos_train_y - train_y_mean)/train_y_std
    sarcos_test_y = (sarcos_test_y - train_y_mean)/train_y_std

    sarcos_train_X = np.array(sarcos_train_X, copy=True, dtype=float, order="C")
    sarcos_train_y = np.array(sarcos_train_y, copy=True, dtype=float, order="C")
    sarcos_test_X = np.array(sarcos_test_X, copy=True, dtype=float, order="C")
    sarcos_test_y = np.array(sarcos_test_y, copy=True, dtype=float, order="C")

    return sarcos_train_X, sarcos_train_y, sarcos_test_X, sarcos_test_y, hyperparams


def main():

    sarcos_dir = os.path.join(os.getenv("SIGVISA_HOME"), "papers", "product_tree", "sarcos")

    sarcos_train_X, sarcos_train_y, sarcos_test_X, sarcos_test_y, hyperparams = load_sarcos(sdir=sarcos_dir)
    np.save(os.path.join(sarcos_dir, "testX.npy"), sarcos_test_X)
    np.save(os.path.join(sarcos_dir, "testy.npy"), sarcos_test_y)
    np.save(os.path.join(sarcos_dir, "hyperparams.npy"), hyperparams)
    print "loaded sarcos data and converted to numpy format"

    train_realdata_gp(sarcos_dir, sarcos_train_X, sarcos_train_y, hyperparams)
    print "trained model"
    test_predict(sarcos_dir)
    print "evaluated predictions"

    eval_gp(bdir=sarcos_dir, test_n=100)
    print "timings finished"

if __name__ == "__main__":
    main()
