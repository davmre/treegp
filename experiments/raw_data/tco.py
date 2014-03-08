import numpy as np
import os
import sys

from synth_dataset import mkdir_p, eval_gp
from real_dataset import train_realdata_gp, test_predict, learn_hyperparams
from sigvisa.models.spatial_regression.SparseGP import SparseGP

def gen_tco(tco_dir):
    f = open(os.path.join(tco_dir, "tco.csv"), 'r')
    X = []
    y = []
    for line in f:
        lon, lat, depth, tco = [float(x) for x in line.split(',')]
        if tco > 0.000001:
            X.append([lon, lat, depth])
            y.append(tco)

    X = np.array(X)
    y = np.array(y)

    p = np.random.permutation(len(y))
    train_n = int(len(y) * 0.2)
    trainX = X[p[:train_n], :]
    trainy = y[p[:train_n]]
    testX = X[p[train_n:], :]
    testy = y[p[train_n:]]

    np.savez(os.path.join(tco_dir, "tco.npz") , trainX=trainX, trainy=trainy, testX=testX, testy=testy, X=X, y=y, p=p)

def load_tco(tco_dir):
    z = np.load(os.path.join(tco_dir, "tco.npz"))
    return z['trainX'], z['trainy'], z['testX'], z['testy']

def main():

    tco_dir = os.path.join(os.getenv("SIGVISA_HOME"), "papers", "product_tree", "tco_learned")

    #gen_tco(tco_dir)
    tco_train_X, tco_train_y, tco_test_X, tco_test_y = load_tco(tco_dir=tco_dir)

    #hyperparams = learn_hyperparams(tco_dir, tco_train_X, tco_train_y, hyperparams=np.array([1, 100, 100, 100], dtype=float), dfn_str='lld', sparse_invert=True, k=1000)
    hyperparams = np.array([95.140702,  12552.9422512, 1257.77376988,  100.], dtype=float)
    #hyperparams = np.array([95.140702,  12552.9422512, 100.0,  100.], dtype=float)

    np.save(os.path.join(tco_dir, "testX.npy"), tco_test_X)
    np.save(os.path.join(tco_dir, "testy.npy"), tco_test_y)
    np.save(os.path.join(tco_dir, "hyperparams.npy"), hyperparams)

    print "loaded data"
    train_realdata_gp(tco_dir, tco_train_X, tco_train_y, hyperparams=hyperparams, dfn_str='lld', sparse_invert=False, basisfns = [lambda x : 1,], param_cov=np.array(((10000,),)), param_mean = np.array((0,)))
    print "trained model"
    test_predict(tco_dir)
    print "evaluated predictions"

    #eval_gp(bdir=tco_dir, test_n=100)
    print "timings finished"

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print e
        import pdb, sys, traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
