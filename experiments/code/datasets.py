import numpy as np
import cPickle as pickle
import os, errno

BASEDIR="experiments/"

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def get_data_dir(dataset_name):
    d = os.path.join(BASEDIR, "datasets", dataset_name)
    mkdir_p(d)
    return d

def test_data(dataset_name):
    data_dir = get_data_dir(dataset_name)

    try:
        X_test = np.loadtxt(os.path.join(data_dir, "X_test.txt"), skiprows=1, delimiter=',')
        y_test = np.loadtxt(os.path.join(data_dir, "y_test.txt"), skiprows=1, delimiter=',')
    except IOError:
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    return X_test, y_test

def training_data(dataset_name, n=None):
    data_dir = get_data_dir(dataset_name)

    try:
        X_train = np.loadtxt(os.path.join(data_dir, "X_train.txt"), skiprows=1, delimiter=',')
        y_train = np.loadtxt(os.path.join(data_dir, "y_train.txt"), skiprows=1, delimiter=',')
    except IOError:
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))

    if n is not None:
        X_train = np.array(X_train[:n,:], copy=True)
        y_train = np.array(y_train[:n], copy=True)

    return X_train, y_train

def load_hparams(dataset_name=None, model_name=None, tag=None, hparam_file=None):
    fname = "hyperparams.pkl" if tag is None else "hyperparams_%s.pkl" % tag
    if hparam_file is None:
        hparam_file = os.path.join(BASEDIR, "models", dataset_name, model_name, fname)

    with open(hparam_file, 'r') as f:
        hparams = pickle.load(f)
    return hparams['cov_main'], hparams['cov_fic'], hparams['noise_var']

def save_hparams(cov_main, cov_fic, noise_var, dataset_name=None, model_name=None, tag=None, hparam_file=None):
    fname = "hyperparams.pkl" if tag is None else "hyperparams_%s.pkl" % tag

    if hparam_file is None:
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

def compiled_tree_fname(dataset_name, model_name, tag):
    tstr = "" if tag is None else "_%s" % tag
    return os.path.join(BASEDIR, "models", dataset_name, model_name, 'tree%s' % tstr)

def timing_results_fname(dataset_name, model_name, tag):
    tstr = "" if tag is None else "_%s" % tag
    return os.path.join(BASEDIR, "models", dataset_name, model_name, 'timings%s.txt' % tstr)

def timing_errors_fname(dataset_name, model_name, tag):
    tstr = "" if tag is None else "_%s" % tag
    return os.path.join(BASEDIR, "models", dataset_name, model_name, 'error%s.npz' % tstr)

def get_spearmint_dir(dataset_name, model_name, tag):
    d = os.path.join(BASEDIR, "spearmint", dataset_name, model_name)
    if tag:
        d = os.path.join(d, tag)
    mkdir_p(d)
    return d
