import os, errno
import sys
import numpy as np
import scipy.linalg
import matplotlib
import itertools
import time
from scikits.sparse.cholmod import cholesky as sp_cholesky
import scipy.sparse
import scipy.special

from treegp.gp import GP, GPCov, prior_sample, sort_morton
from treegp.experiments.code.datasets import save_hparams, get_data_dir

basedir = "experiments/datasets/"
#lscales = (0.00005, 0.0005, 0.005, 0.01, 0.02)
#lscales = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 10)
vs = (1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 200.0, 500.0, 1000.0)
#cluster_ns = (50, 100, 200, 400, 1000, 1500, 2000, 2500, 3000, 4000, 8000)
#cluster_ns = (50, 100, 200, 400)
#cluster_ns = (8000,)

#Ns = (1000, 2000, 4000, 8000, 20000, 40000, 80000, 160000) #,50000, 60000, 80000, 160000)#
#Ns = (1000, 2000, 4000, 8000,16000, 24000, 32000, 48000, 64000,) #,50000, 60000, 80000, 160000)
#Ns = (20000, 40000, 60000, 80000)


def genX_clusters(dim, n_clusters, cluster_pts, cluster_width):
    centers = [ np.array([np.random.rand() for d in range(dim)]) for i in range(n_clusters)] # generate n_clusters points within the unit cube of dimension dim
    pts = [  c +  np.array([np.random.randn() * cluster_width for d in range(dim)])  for (c,i) in itertools.product(centers, range(cluster_pts))]
    return np.array(pts)

def sort_events(X):
    X_sorted = np.array(sorted(X, key = lambda x: x[0]), dtype=float)
    return X_sorted

def genX(dim, npts):
    pts = np.array([[np.random.rand() for d in range(dim)] for i in range(npts)]) # generate n_clusters points within the unit cube of dimension dim
    return pts


def scatter2d(X, y, fname):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    f = Figure()
    ax = f.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=2, edgecolors='none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def build_cluster_benchmark():
    sigma2_n = 1.0
    sigma2_f = 1.0
    npts = 10000

    cluster_args = dict()
    cluster_args['n_clusters'] = 100
    lengthscale = np.sqrt(2.0/100.0) / np.sqrt(np.pi)

    for cluster_width in (0.0001, 0.001, 0.003, 0.01, 0.03, 0.1):
        dsetname = "synth_cluster_%f_%d" % (cluster_width, npts)
        cluster_args['cluster_width'] = cluster_width

        create_bench(dsetname=dsetname, dim=2, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f, cluster_args=cluster_args)

def build_lscale_benchmark():
    sigma2_n = 1.0
    sigma2_f = 1.0
    npts=5000

    for points_within_lscale in vs:
        lengthscale = np.sqrt(points_within_lscale/npts) / np.sqrt(np.pi)

        dsetname = "synth_lscale_%d_%d" % (points_within_lscale, npts)
        create_bench(dsetname=dsetname, dim=2, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)

def build_highdim_benchmark():
    sigma2_n = 1.0
    sigma2_f = 1.0
    npts=5000
    points_within_lscale=200.0

    for dims in (2,3,4,5,8,10):
        lengthscale = (points_within_lscale/npts  * scipy.special.gamma(dims/2+1) / np.pi**(dims/2))**(1.0/dims)

        dsetname = "synth_dims_%d_%d_%d" % (dims, points_within_lscale, npts)
        create_bench(dsetname=dsetname, dim=dims, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)


def run_lscale_benchmark():
    npts=1000

    for lengthscale in lscales:


        dsetname = "synth_scale_%.2f_%d" % (lengthscale,npts)
        print "building", npts,  "at", lengthscale


        bdir = os.path.join(basedir, "synth_lscale_%s_%f_%d" % (wfn_str,lengthscale,npts))
        eval_gp(bdir)


def create_bench(dsetname, dim, test_n, npts, lengthscale, sigma2_n, sigma2_f, wfn_str="compact2", cluster_args=None):
    sparse_threshold=1e-8

    cov_main = GPCov(dfn_str="euclidean", wfn_str=wfn_str, wfn_params=[sigma2_f,], dfn_params=[lengthscale,lengthscale])
    save_hparams(dataset_name=dsetname, model_name=wfn_str, cov_main=cov_main, cov_fic=None, noise_var=sigma2_n)


    if cluster_args is not None:
        cluster_args['cluster_pts'] = int((npts+test_n)/cluster_args['n_clusters'])
        X = genX_clusters(dim, **cluster_args)
    else:
        X = genX(dim, npts + test_n)
    test_p = np.random.permutation(len(X))
    test_X = X[test_p[:test_n], :]
    test_y = np.random.randn(test_X.shape[0])

    X = np.array(X[test_p[test_n:], :])
    y = np.random.randn(X.shape[0])


    data_dir = get_data_dir(dsetname)
    np.save(os.path.join(data_dir, 'X_test.npy'), test_X)
    np.save(os.path.join(data_dir, 'y_test.npy'), test_y)


    np.save(os.path.join(data_dir, 'X_train.npy'), X)
    np.save(os.path.join(data_dir, 'y_train.npy'), y)

    """
    gp = GP(X=X, y=y, basisfns = [],
                  dfn_str="euclidean", wfn_str=wfn_str,
                  hyperparams = hyperparams,
                  K = spK, sparse_threshold=sparse_threshold,
                  sort_events=False, build_tree=False)


    gp.save_trained_model(os.path.join(data_dir, 'trained.gp'))
    """

def build_size_benchmark(vs = (1.0,), Ns = (8000,)):
    sigma2_n = 1.0
    sigma2_f = 1.0
    #extra_nc = test_n / cluster_size

    for npts in Ns:
        for points_within_lscale in vs:
            lengthscale = np.sqrt(points_within_lscale/npts) / np.sqrt(np.pi)
            dsetname = "synth_size_%.2fpts_%d" % (points_within_lscale,npts)
            print "building", npts, points_within_lscale, "at", lengthscale
            print dsetname
            create_bench(dsetname=dsetname, dim=2, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)

def build_highd_benchmark():
    sigma2_n = 1.0
    sigma2_f = 1.0
    npts = 10000
    points_within_lscale=1.0
    import scipy.misc
    for dim in (2, 6, 12, 20):
        k = dim/2
        lengthscale = (scipy.misc.factorial(k) * points_within_lscale/npts) ** (.5/k) / np.sqrt(np.pi)
        bdir = os.path.join(basedir, "highd_%d_%s_base%f_%d" % (dim, wfn_str,points_within_lscale,npts))
        if not os.path.exists(bdir):
            create_bench(bdir=bdir, dim=dim, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)



if __name__ == "__main__":
    Ns = (8000, 16000, 24000, 32000, 48000, 64000, 80000, 96000, 128000)

    #vs = (1.0, 5.0, 10.0)
    #vs = (20.0,)

    #build_lscale_benchmark()
    #build_highdim_benchmark()
    build_cluster_benchmark()
    #run_lscale_benchmark()
    #build_size_benchmark(vs = vs, Ns=Ns)
    #run_size_benchmark()

    #main()
