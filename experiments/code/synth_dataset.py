import os, errno
import sys
import numpy as np
import scipy.linalg
import matplotlib
import itertools
import time
from scikits.sparse.cholmod import cholesky as sp_cholesky
import scipy.sparse

from sigvisa.models.spatial_regression.SparseGP import SparseGP, prior_sample

basedir = os.path.join(os.getenv("SIGVISA_HOME"), "papers", "product_tree", "benchmarks")
#lscales = (0.00005, 0.0005, 0.005, 0.01, 0.02)
lscales = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 10)
#cluster_ns = (50, 100, 200, 400, 1000, 1500, 2000, 2500, 3000, 4000, 8000)
#cluster_ns = (50, 100, 200, 400)
#cluster_ns = (8000,)

#Ns = (1000, 2000, 4000, 8000, 20000, 40000, 80000, 160000) #,50000, 60000, 80000, 160000)#
#Ns = (1000, 2000, 4000, 8000,16000, 24000, 32000, 48000, 64000,) #,50000, 60000, 80000, 160000)
#Ns = (20000, 40000, 60000, 80000)
Ns = (8000,)

wfn_str = "se"

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


def eval_gp(bdir=None, gp=None, testX=None, resultfile=None, errorfile=None, test_n=None, cutoff_rule=1, flag=""):
    import scipy.stats

    gp = SparseGP(fname=os.path.join(bdir, 'trained.gp')) if gp is None else gp
    testX = np.load(os.path.join(bdir, "testX.npy")) if testX is None else testX
    resultfile = os.path.join(bdir, 'results_cutoff%d%s.txt' % (cutoff_rule,flag)) if resultfile is None else resultfile
    errorfile = os.path.join(bdir, "error.npz") if errorfile is None else errorfile


    if test_n is None:
        test_n = len(testX)

    naive_predict = np.zeros(test_n)
    naive_predict_times = np.zeros(test_n)

    tree_predict = np.zeros(test_n)
    tree_predict_terms = np.zeros(test_n, dtype=int)
    tree_predict_times = np.zeros(test_n)

    dense_covar = np.zeros(test_n)
    dense_covar_times = np.zeros(test_n)

    sparse_covar = np.zeros(test_n)
    sparse_covar_times = np.zeros(test_n)
    sparse_covar_qftimes = np.zeros(test_n)
    sparse_covar_nonqftimes = np.zeros(test_n)

    sparse_covar_spkernel = np.zeros(test_n)
    sparse_covar_spkernel_times = np.zeros(test_n)
    sparse_covar_spkernel_qftimes = np.zeros(test_n)
    sparse_covar_spkernel_nonqftimes = np.zeros(test_n)

    sparse_covar_spkernel_solve = np.zeros(test_n)
    sparse_covar_spkernel_solve_times = np.zeros(test_n)

    for i in range(test_n):
        t0 = time.time()
        naive_predict[i] = gp.predict_naive(testX[i:i+1,:])
        t1 = time.time()
        naive_predict_times[i] = t1-t0


    for i in range(test_n):
        t2 = time.time()
        tree_predict[i] = gp.predict(testX[i:i+1,:], eps=1e-4)
        t3 = time.time()
        tree_predict_times[i] = t3-t2
        tree_predict_terms[i] = gp.predict_tree.fcalls


    for i in range(test_n):
        t4 = time.time()
        sparse_covar[i] = gp.covariance(testX[i:i+1,:])
        t5 = time.time()
        sparse_covar_times[i] = t5-t4
        sparse_covar_qftimes[i] = gp.qf_time
        sparse_covar_nonqftimes[i] = gp.nonqf_time



    for i in range(test_n):
        t41 = time.time()
        sparse_covar_spkernel[i] = gp.covariance_spkernel(testX[i:i+1,:])
        t51 = time.time()
        sparse_covar_spkernel_times[i] = t51-t41
        sparse_covar_spkernel_qftimes[i] = gp.qf_time
        sparse_covar_spkernel_nonqftimes[i] = gp.nonqf_time

    for i in range(test_n):
        t41 = time.time()
        sparse_covar_spkernel_solve[i] = gp.covariance_spkernel_solve(testX[i:i+1,:])
        t51 = time.time()
        sparse_covar_spkernel_solve_times[i] = t51-t41


    has_dense = True
    try:
        for i in range(100):
            t42 = time.time()
            dense_covar[i] = gp.covariance_dense(testX[i:i+1,:])
            t52 = time.time()
            dense_covar_times[i] = t52-t42
    except:
        has_dense=False


    def strstats(v):
        if v.dtype == float:
            return "mean %f std %f min %f 10th %f 50th %f 90th %f max %f" % stats(v)
        elif v.dtype == int:
            return "mean %d std %d min %d 10th %d 50th %d 90th %d max %d" % stats(v)
        else:
            raise Exception("weird dtype %s"% v.dtype)

    def stats(v):
      return (np.mean(v), np.std(v), np.min(v), scipy.stats.scoreatpercentile(v, 10), np.median(v) ,scipy.stats.scoreatpercentile(v, 90) , np.max(v))

    f = open(resultfile, 'w')

    best_mean_time = np.float("inf")
    best_ei = -1
    best_ej = -1


    if cutoff_rule == 0:
        eps_rels =  (2,4,8,16)
        eps_abses = (1,)
    elif cutoff_rule == 1:
        eps_rels =  (2,4,8,16)
        eps_abses = (4,5,6,8)
    elif cutoff_rule == 2:
        eps_rels = (4,)
        eps_abses = (1,2,3,)

    tree_covar = np.zeros((len(eps_rels), len(eps_abses), test_n))
    tree_covar_terms = np.zeros((len(eps_rels), len(eps_abses), test_n), dtype=int)
    tree_covar_distevals = np.zeros((len(eps_rels), len(eps_abses), test_n), dtype=int)
    tree_covar_times = np.zeros((len(eps_rels), len(eps_abses), test_n))
    tree_covar_qftimes = np.zeros((len(eps_rels), len(eps_abses), test_n))
    tree_covar_nonqftimes = np.zeros((len(eps_rels), len(eps_abses), test_n))
    for (e_i, epsm) in enumerate(eps_rels):
        for (e_j, eps_abs) in enumerate(eps_abses):
            for i in range(test_n):
                t6 = time.time()
                tree_covar[e_i, e_j, i] = gp.covariance_double_tree(testX[i:i+1,:], eps=10**(-epsm), eps_abs=10**(-eps_abs), cutoff_rule=cutoff_rule)
                t7 = time.time()
                tree_covar_terms[e_i, e_j, i] = gp.double_tree.fcalls
                tree_covar_distevals[e_i, e_j, i] = gp.double_tree.dfn_evals
                tree_covar_times[e_i, e_j, i] = t7-t6
                tree_covar_qftimes[e_i, e_j, i] = gp.qf_time
                tree_covar_nonqftimes[e_i, e_j, i] = gp.nonqf_time
                #print
                #print

            if np.mean(np.abs((tree_covar[e_i, e_j, :] - sparse_covar)/sparse_covar)) < 0.001:
                mean_time = np.mean(tree_covar_times)
                if mean_time < best_mean_time:
                    best_mean_time = mean_time
                    best_ei = e_i
                    best_ej = e_j

            f.write("tree covar%d_%d times: %s\n" % (epsm, eps_abs, strstats(tree_covar_times[e_i, e_j, :])))
            f.write("tree covar%d_%d qftimes: %s\n" % (epsm, eps_abs, strstats(tree_covar_qftimes[e_i, e_j, :])))
            f.write("tree covar%d_%d nqftimes: %s\n" % (epsm, eps_abs, strstats(tree_covar_nonqftimes[e_i, e_j, :])))
            f.write("tree covar%d_%d terms: %s\n" % (epsm, eps_abs, strstats(tree_covar_terms[e_i, e_j, :])))
            f.write("tree covar%d_%d dfnevals: %s\n" % (epsm, eps_abs, strstats(tree_covar_distevals[e_i, e_j, :])))
            f.write("tree covar%d_%d rel errors: %s \n" %  (epsm, eps_abs, strstats(np.abs((tree_covar[e_i, e_j, :] - sparse_covar)/(1-sparse_covar)))))
            f.write("tree covar%d_%d var-rel errors: %s \n" %  (epsm, eps_abs, strstats(np.abs((tree_covar[e_i, e_j, :] - sparse_covar)/sparse_covar))))
            f.write("tree covar%d_%d abs errors: %s \n" %  (epsm, eps_abs, strstats(np.abs(tree_covar[e_i, e_j, :] - sparse_covar))))
            f.write("\n")
            f.flush()

    f.write("naive predict times: %s\n" % strstats(naive_predict_times))
    f.write("\n")
    f.write("tree predict times: %s\n" % strstats(tree_predict_times))
    f.write("tree predict terms:  %s\n" % strstats(tree_predict_terms))
    f.write("tree predict errors: %s\n" % strstats(np.abs(tree_predict - naive_predict)))
    f.write("\n")

    if has_dense:
        f.write("dense covar times: %s\n" % strstats(dense_covar_times))
    f.write("sparse covar times: %s\n" % strstats(sparse_covar_times))
    f.write("sparse covar qftimes: %s\n" % strstats(sparse_covar_qftimes))
    f.write("sparse covar nqftimes: %s\n" % strstats(sparse_covar_nonqftimes))
    f.write("\n")
    f.write("sparse covar spkernel times: %s\n" % strstats(sparse_covar_spkernel_times))
    f.write("sparse covar spkernel qftimes: %s\n" % strstats(sparse_covar_spkernel_qftimes))
    f.write("sparse covar spkernel nqftimes: %s\n" % strstats(sparse_covar_spkernel_nonqftimes))
    f.write("sparse covar spkernel error: %f\n" % np.mean(np.abs(sparse_covar_spkernel_solve - sparse_covar)))
    f.write("\n")
    f.write("sparse covar spkernel_solve times: %s\n" % strstats(sparse_covar_spkernel_solve_times))
    f.write("sparse covar spkernel_solve error: %f\n" % np.mean(np.abs(sparse_covar_spkernel_solve - sparse_covar)))
    f.write("\n")

    f.write("actual vars: %s\n" % strstats(sparse_covar))
    f.write("actual qfs: %s\n" % strstats(1-sparse_covar))
    f.write("\n")

    e_i = best_ei
    e_j = best_ej

    if e_i == -1 or e_j == -1:
        f.write('product tree failed! no param setting returned accurate variances.\n')
    else:
        epsm = eps_rels[e_i]
        eps_abs = eps_abses[e_j]
        f.write("best tree: covar%d_%d times: %s\n" % (epsm, eps_abs, strstats(tree_covar_times[e_i, e_j, :])))
        f.write("best tree: covar%d_%d qftimes: %s\n" % (epsm, eps_abs, strstats(tree_covar_qftimes[e_i, e_j, :])))
        f.write("best tree: covar%d_%d nqftimes: %s\n" % (epsm, eps_abs, strstats(tree_covar_nonqftimes[e_i, e_j, :])))
        f.write("best tree: covar%d_%d terms: %s\n" % (epsm, eps_abs, strstats(tree_covar_terms[e_i, e_j, :])))
        f.write("best tree: covar%d_%d dfnevals: %s\n" % (epsm, eps_abs, strstats(tree_covar_distevals[e_i, e_j, :])))
        f.write("best tree: covar%d_%d rel errors: %s \n" %  (epsm, eps_abs, strstats(np.abs((tree_covar[e_i, e_j, :] - sparse_covar)/(1-sparse_covar)))))
        f.write("best tree: covar%d_%d var-rel errors: %s \n" %  (epsm, eps_abs, strstats(np.abs((tree_covar[e_i, e_j, :] - sparse_covar)/sparse_covar))))
        f.write("best tree: covar%d_%d abs errors: %s \n" %  (epsm, eps_abs, strstats(np.abs(tree_covar[e_i, e_j, :] - sparse_covar))))
        f.write("\n")

    f.close()

    np.savez(errorfile, tree_covar=tree_covar, sparse_covar=sparse_covar, tree_predict=tree_predict, naive_predict=naive_predict, sparse_covar_spkernel=sparse_covar_spkernel)


"""
def geny(X, cluster_width, sigma2_n):
    # sample from a GP prior with SE kernel

    n = X.shape[0]
    print "starting y"
    K, spK = kernel(X, cluster_width, sigma2_n)
    print "got kernel"
    z = np.random.randn(n)

    t0  = time.time()
    factor = sp_cholesky(spK)
    t1 = time.time()
    L = factor.L()
    P = factor.P()
    Pinv = np.argsort(P)

    t2 = time.time()
    y = (L * z)[Pinv]
    t3 = time.time()
    print "chol on %d elements took time %.2f" % (n, t1-t0)
    print "mult on %d elements took time %.2f" % (n, t3-t2)
    return y
"""

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

def build_lscale_benchmark():
    sigma2_n = 0.000001
    sigma2_f = 1.0
    npts=1000

    for lengthscale in lscales:
        bdir = os.path.join(basedir, "%s_%f_%d" % (wfn_str,lengthscale,npts))
        create_bench(bdir=bdir, dim=2, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)

def run_lscale_benchmark():
    npts=1000

    for lengthscale in lscales:
        bdir = os.path.join(basedir, "%s_%f_%d" % (wfn_str,lengthscale,npts))
        eval_gp(bdir)


def create_bench(bdir, dim, test_n, npts, lengthscale, sigma2_n, sigma2_f):
    mkdir_p(bdir)
    sparse_threshold=1e-8
    hyperparams = [sigma2_n, sigma2_f, lengthscale, lengthscale]
    X = genX(dim, npts + test_n)
    test_p = np.random.permutation(len(X))
    test_X = X[test_p[:test_n], :]
    X = np.array(X[test_p[test_n:], :])

    X = sort_events(X)

    y, spK = prior_sample(X, hyperparams=hyperparams, dfn_str="euclidean", wfn_str=wfn_str, sparse_threshold=sparse_threshold, return_kernel=True)

    np.save(os.path.join(bdir, 'testX.npy'), test_X)
    np.save(os.path.join(bdir, 'hyperparams.npy'), hyperparams)
    np.save(os.path.join(bdir, 'K.npy'), spK)
    np.save(os.path.join(bdir, 'X.npy'), X)
    np.save(os.path.join(bdir, 'y.npy'), y)

    gp = SparseGP(X=X, y=y, basisfns = [],
                  dfn_str="euclidean", wfn_str=wfn_str,
                  hyperparams = hyperparams,
                  K = spK, sparse_threshold=sparse_threshold,
                  sort_events=False, build_tree=False)


    gp.save_trained_model(os.path.join(bdir, 'trained.gp'))

    with open(os.path.join(bdir, 'stats.txt'), 'w') as f:
        f.write("hyperparams: %s\n" % hyperparams)
        nzr, nzc = spK.nonzero()
        f.write("K nonzero: %d\n" % len(nzr))

        nzr, nzc = gp.Kinv.nonzero()
        f.write("Kinv nonzero: %d\n" % len(nzr))
        for k in gp.timings:
            f.write("timing %s: %.3f\n" % (k, gp.timings[k]))


def build_size_benchmark():
    sigma2_n = 1.0
    sigma2_f = 1.0
    #extra_nc = test_n / cluster_size

    for npts in Ns:
        for points_within_lscale in (0.25, 1.0, 5.0, ):
            lengthscale = np.sqrt(points_within_lscale/npts) / np.sqrt(np.pi)
            bdir = os.path.join(basedir, "%s_%.2fpts_%d" % (wfn_str,points_within_lscale,npts))
            if not os.path.exists(os.path.join(bdir, "trained.gp")):
                print "building", npts, points_within_lscale, "at", lengthscale
                print bdir
                create_bench(bdir=bdir, dim=2, test_n=1000, npts=npts, lengthscale=lengthscale, sigma2_n=sigma2_n, sigma2_f=sigma2_f)

def run_size_benchmark():
    for npts in Ns:
        for points_within_lscale in (1.0,5.0):
            bdir = os.path.join(basedir, "%s_%.2fpts_%d" % (wfn_str,points_within_lscale,npts))
            print bdir
            eval_gp(bdir)

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



def main():

    sigma2_n = 0.000001
    lengthscale = 0.000005
    sigma2_f = 1.0
    hyperparams = [sigma2_n, sigma2_f, lengthscale, lengthscale]

    X = genX(20, 2000, 20, lengthscale)
    print "got X"
    y, spK = prior_sample(X, hyperparams=hyperparams, dfn_str="euclidean", wfn_str=wfn_str, return_kernel=True)
    print "got y"

    gp = SparseGP(X=X, y=y, basisfns = [],
                  kernel_str="euclidean_se",
                  hyperparams = hyperparams,
                  K = spK,
                  sort_events=False)

    ymax = np.max(y)
    ymin = np.min(y)
    normy = (y - ymin) / (ymax-ymin)

    scatter2d(X, normy, 'plot.png')
    print "scattered"

    recovered = gp.predict(X)
    normr = (recovered - ymin) / (ymax - ymin)
    np.savetxt('recovered.txt', recovered)
    scatter2d(X, normr, 'recovered.png')

    print "covar", gp.covariance(X[0:1, :])

if __name__ == "__main__":
    #build_lscale_benchmark()
    #run_lscale_benchmark()
    build_size_benchmark()
    #run_size_benchmark()
    #build_highd_benchmark()
    #main()
