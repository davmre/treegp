import numpy as np
import cPickle as pickle
import os, errno

BASEDIR="/home/dmoore/python/sparsegp/experiments/"

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def test_data(dataset_name):
    data_dir = os.path.join(BASEDIR, "datasets", dataset_name)
    X_test = np.loadtxt(os.path.join(data_dir, "X_test.txt"), skiprows=1, delimiter=',')
    y_test = np.loadtxt(os.path.join(data_dir, "y_test.txt"), skiprows=1, delimiter=',')

    return X_test, y_test

def training_data(dataset_name, n=None):
    data_dir = os.path.join(BASEDIR, "datasets", dataset_name)

    X_train = np.loadtxt(os.path.join(data_dir, "X_train.txt"), skiprows=1, delimiter=',')
    y_train = np.loadtxt(os.path.join(data_dir, "y_train.txt"), skiprows=1, delimiter=',')

    if n is not None:
        X_train = np.array(X_train[:n,:], copy=True)
        y_train = np.array(y_train[:n], copy=True)

    return X_train, y_train

def load_hparams(dataset_name, model_name, tag=None):
    fname = "hyperparams.pkl" if tag is None else "hyperparams_%s.pkl" % tag
    hparam_file = os.path.join(BASEDIR, "models", dataset_name, model_name, fname)
    with open(hparam_file, 'r') as f:
        hparams = pickle.load(f)
    return hparams['cov_main'], hparams['cov_fic'], hparams['noise_var']

def save_hparams(dataset_name, model_name, cov_main, cov_fic, noise_var, tag=None):
    fname = "hyperparams.pkl" if tag is None else "hyperparams_%s.pkl" % tag
    hparam_dir = os.path.join(BASEDIR, "models", dataset_name, model_name)
    mkdir_p(hparam_dir)

    hparam_file = os.path.join(hparam_dir, fname)

    hparams = dict()
    hparams['cov_main']=cov_main
    hparams['cov_fic']=cov_fic
    hparams['noise_var']=noise_var
    with open(hparam_file, 'w') as f:
        pickle.dump(hparams, f)

def gp_fname(dataset_name, model_name, n=None, tag=None):
    tstr = "" if tag is None else "_%s" % tag
    fname = "trained%s.gp" % tstr if n is None else "trained%s_%d.gp" % (tstr, n)
    return os.path.join(BASEDIR, "models", dataset_name, model_name, fname)

def predict_results_fname(dataset_name, model_name, tag):
    tstr = "" if tag is None else "_%s" % tag
    return os.path.join(BASEDIR, "models", dataset_name, model_name, 'accuracy%s.txt' % tstr)

def timing_results_fname(dataset_name, model_name, tag):
    tstr = "" if tag is None else "_%s" % tag
    return os.path.join(BASEDIR, "models", dataset_name, model_name, 'timings%s.txt' % tstr)

def timing_errors_fname(dataset_name, model_name, tag):
    tstr = "" if tag is None else "_%s" % tag
    return os.path.join(BASEDIR, "models", dataset_name, model_name, 'error%s.npz' % tstr)
