import os
import sys
import numpy as np
import time

from cross_validation import save_cv_folds, train_cv_models, cv_eval_models
from synth_dataset import mkdir_p, eval_gp
from sigvisa.models.spatial_regression.SparseGP import SparseGP
from sigvisa.models.spatial_regression.baseline_models import LinearBasisModel
import itertools

base1 = os.path.join(os.getenv("SIGVISA_HOME"), "papers", "product_tree")
wfn_str = "se"

def plot_distance(X, y, model, axes, nstd, azi=0, depth=0):

    axes.scatter(X[:, 3], y, alpha = 6/np.log(len(y)), s=15, marker='.', edgecolors="none", c="red")

    #pred = model.predict(X)
    distances = np.linspace(0, 10000, 100)
    fakeX = np.array([[0, 0, 0, dist, 0] for dist in distances])
    pred = model.predict(fakeX, parametric_only=True)

    axes.plot(distances, pred, 'k-')

    stds = np.zeros((len(distances),))
    for i in range(len(fakeX)):
        v = model.variance(fakeX[i:i+1, :], parametric_only=True, include_obs=True)
        stds[i] = np.sqrt(v)
    std = np.array(stds) + nstd

    var_x = np.concatenate((distances, distances[::-1]))
    var_y = np.concatenate((pred + 2 * std, (pred - 2 * std)[::-1]))
    axes.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3', alpha=0.4)

def setup_azi_basisfns(order, n_azi_buckets, param_var):

    if n_azi_buckets == 0:
        basisfn_strs = ["lambda x : " + ("1" if d==0 else "(x[3]/1000.0)**%d" % d)   for d in range(order+1)]
    else:
        azi_boundaries = np.linspace(0, 360, n_azi_buckets+1)
        basisfn_strs = ["lambda x : " + ("1" if d==0 else "(x[3]/1000.0)**%d" % d) + " if %f <= x[4] < %f else 0" % (azi_boundaries[azi_i], azi_boundaries[azi_i+1])  for (d, azi_i) in itertools.product(range(order+1), range(len(azi_boundaries)-1))]

    basisfns = [eval(s) for s in basisfn_strs]

    k = len(basisfns)
    b = np.zeros((k,))
    #vars = np.array([param_var ** ((-1) * d) for d in range(k)])
    vars = np.array([param_var  for d in range(k)])
    for i in range(0, k, order+1):
        vars[i] = (1000000)**2
    #vars[1] = (10)**2
    B = np.diag(vars)
    return basisfns, b, B

def get_nstd(X, y, order, n_azi_buckets, param_var):
    basisfns, b, B = setup_azi_basisfns(order, n_azi_buckets, param_var)
    lbm1 = LinearBasisModel(X=X, y=y, basisfns = basisfns, param_mean=b, param_covar=B, noise_std=1.0, sta="FITZ")
    pred = lbm1.predict(X)
    nstd = np.std(y-pred)
    return nstd

def plot_gp(X, y, nstd, lscale, order, n_azi_buckets, param_var, cv_dir):
    basisfns, b, B = setup_azi_basisfns(order, n_azi_buckets, param_var)
    sgp = SparseGP(X=X, y=y, basisfns = basisfns, param_mean=b, param_cov=B, hyperparams=[nstd, 1.0, lscale, lscale], sta="FITZ", dfn_str="lld", wfn_str=wfn_str)
    sgp.save_trained_model(os.path.join(cv_dir, "fitz%d_gp%d_%d.sgp" % (lscale, order, n_azi_buckets)))

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    f = Figure()
    ax = f.add_subplot(111)
    plot_distance(X, y, sgp, ax, nstd)
    ax.set_xlim([1900, 10000])
    ax.set_ylim([-2, 4])
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(os.path.join(cv_dir, "gp%d_%d_%d.png" % (lscale, order, n_azi_buckets)), bbox_inches='tight')


def cov_timing(cv_dir, lscale, order, n_azi_buckets):
    sgp = SparseGP(fname=os.path.join(cv_dir, "fold_00.gp%d_%d_%d" % (lscale, order, n_azi_buckets)))
    X = np.loadtxt(os.path.join(cv_dir, "X.txt"))
    test_idx = np.array([int(z) for z in np.loadtxt(os.path.join(cv_dir, "fold_00_test.txt"))])
    testX = X[test_idx]
    resultfile = os.path.join(cv_dir, "results_gp%d_%d_%d.txt" % (lscale, order, n_azi_buckets))
    errorfile = os.path.join(cv_dir, "error_gp%d_%d_%d.npz" % (lscale, order, n_azi_buckets))
    eval_gp(gp=sgp, testX=testX, resultfile=resultfile, errorfile=errorfile)

    poly = LinearBasisModel(fname=os.path.join(cv_dir, "fold_00.poly_%d_%d" % (order, n_azi_buckets)))
    resultfile_poly = os.path.join(cv_dir, "results_poly_%d_%d.txt" % (order, n_azi_buckets))
    f = open(resultfile_poly, 'w')
    test_n = len(test_idx)
    poly_covars = np.zeros((test_n,))
    t0 = time.time()
    for i in range(test_n):
        poly_covars[i] = poly.covariance(testX[i:i+1,:])
    t1 = time.time()
    f.write("cov time %f\n" % ((t1-t0)/test_n))
    f.close()

def main():

    #order=0
    #lscale = 10
    # example python fitx.py LPAZ amp lscale order
    sta = sys.argv[1]
    target = sys.argv[2]
    lscale = int(sys.argv[3])
    order = int(sys.argv[4])
    n_azi_buckets = int(sys.argv[5])
    param_var=10

    X = np.loadtxt(os.path.join(base1, "X_%s_%s.txt" % (target, sta)))
    y = np.loadtxt(os.path.join(base1, "y_%s_%s.txt" % (target, sta)))

    good_idx = X[:, 2] < 15
    X = X[good_idx, :]
    y = y[good_idx]

    cv_dir = os.path.join(base1, "cv_%s_%s" % (target, sta))
    mkdir_p(cv_dir)

    nstd = get_nstd(X, y, order, n_azi_buckets, param_var)
    plot_gp(X, y, nstd, lscale, order, n_azi_buckets, param_var, cv_dir)
    #cv_gp(X, y, nstd, lscale, order, n_azi_buckets, param_var, cv_dir)
    #cv_poly(X, y, nstd, order, n_azi_buckets, param_var, cv_dir)
    #cov_timing(cv_dir, lscale, order, n_azi_buckets)

def cv_poly(X, y, nstd, order, n_azi_buckets, param_var, cv_dir):
    load_model = lambda fname : LinearBasisModel(fname=fname)
    basisfns, b, B = setup_azi_basisfns(order, n_azi_buckets, param_var=param_var)
    learn_model = lambda X, y: LinearBasisModel(X=X, y=y, basisfns = basisfns, param_mean=b, param_covar=B, noise_std=nstd, sta="FITZ")
    do_cv(X=X, y=y, model_type="poly_%d_%d" % (order, n_azi_buckets), learn_model=learn_model, load_model=load_model, cv_dir=cv_dir)

def cv_gp(X, y, nstd, lscale, order, n_azi_buckets, param_var, cv_dir):
    load_model = lambda fname : SparseGP(fname=fname)
    basisfns, b, B = setup_azi_basisfns(order, n_azi_buckets, param_var=param_var)
    learn_model = lambda X, y: SparseGP(X=X, y=y, basisfns = basisfns, param_mean=b, param_cov=B, hyperparams=[nstd, 1.0, lscale, lscale], sta="FITZ", dfn_str="lld", wfn_str=wfn_str)
    do_cv(X=X, y=y, model_type="gp%d_%d_%d" % (lscale, order, n_azi_buckets), learn_model=learn_model, load_model=load_model, cv_dir=cv_dir)

def do_cv(X, y, model_type, learn_model, load_model, cv_dir):
    evids = range(len(y))
    np.savetxt(os.path.join(cv_dir, "X.txt"), X)
    np.savetxt(os.path.join(cv_dir, "y.txt"), y)
    np.savetxt(os.path.join(cv_dir, "evids.txt"), evids)

    print "generating cross-validation folds in dir", cv_dir
    save_cv_folds(X, y, evids, cv_dir, folds=5)

    train_cv_models(cv_dir, learn_model,  model_type)

    print "DONE TRAINING, NOW EVALUATING!"
    mean_error, median_error = cv_eval_models(cv_dir, load_model=load_model, model_type=model_type)


if __name__ == "__main__":
    #build_lscale_benchmark()
    #run_lscale_benchmark()
    main()
