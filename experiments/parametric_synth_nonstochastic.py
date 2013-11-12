from sigvisa.models.spatial_regression.SparseGP import SparseGP
from sigvisa.models.spatial_regression.baseline_models import LinearBasisModel
from sigvisa.utils.cover_tree import VectorTree
import pyublas
import sys
import scipy.linalg
import numpy as np
import time

def savescatter(X, y, fname):
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

def save_surface(fn, fname, n=100, **kwargs):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    f = Figure((6,6))
    ax = f.add_subplot(111)

    x = y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n,n))
    t0 = time.time()
    for i in range(n):
        for j in range(n):
            Z[i,j] = fn(x[i], y[j])
    t1 = time.time()
    print "generated", fname, "in", t1-t0

    ax.pcolor(X, Y, Z.T, **kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname, bbox_inches='tight')



n = 1000
test_n = 500

X = np.random.rand(n+test_n,2)

long_scale=.5
short_scale=.02
sigma2_n = 0.1
sigma2_f_long = 5.0
sigma2_f_short = 1.0

dfn_params_long=np.array([long_scale, long_scale], dtype=float)
dfn_params_short=np.array([short_scale,short_scale], dtype=float)
wfn_params_long = np.array([sigma2_f_long,], dtype=float)
wfn_params_short = np.array([sigma2_f_short,], dtype=float)

long_tree = VectorTree(X[0:1,:], 1, "euclidean", dfn_params_long, "se", wfn_params_long)
short_tree = VectorTree(X[0:1,:], 1, "euclidean", dfn_params_short, "se", wfn_params_short)

def k(X1, X2, identical=False):
    long_kernel = long_tree.kernel_matrix(X1, X2, False)
    short_kernel = short_tree.kernel_matrix(X1, X2, False)
    kernel= long_kernel + short_kernel
    if identical:
        kernel += sigma2_n * np.eye(len(X1))
    return kernel

kernel = k(X, X, True)
L = scipy.linalg.cholesky(kernel, lower=True)
z = np.random.randn(n+test_n)
y = np.dot(L, z)

test_X = X[n:, :]
X = X[:n,:]
test_y = y[n:]
y = y[:n]


import itertools
def poly_basisfns_2d(order):
    basisfn_strs = ["lambda x : " + ("x[0]**%d * x[1]**%d" % (d1, d2)) for (d1, d2) in itertools.product(range(order+1), range(order+1))]
    return [eval(s) for s in basisfn_strs]
bfn = poly_basisfns_2d(5)
d = len(bfn)
b = np.zeros((d,))
B = np.eye(d) * 10
print bfn

hyperparams = [sigma2_n, sigma2_f_short, short_scale, short_scale]
sgp  = SparseGP(X=X, y=y, basisfns = bfn, param_mean=b, param_cov=B, dfn_str="euclidean", wfn_str="se", hyperparams=hyperparams, sort_events=False)

sgp_plain  = SparseGP(X=X, y=y, basisfns = [], param_mean=None, param_cov=None, dfn_str="euclidean", wfn_str="se", hyperparams=hyperparams, sort_events=False)

poly_model = LinearBasisModel(X=X, y=y, basisfns=bfn, param_mean=b, param_covar=B, noise_std=np.sqrt(sigma2_n))


npy = np.reshape(sgp_plain.predict(X), (-1,))
py = np.reshape(sgp.predict(X, parametric_only=True), (-1,))
p2y = np.reshape(poly_model.predict(X), (-1,))
gy =  np.reshape(sgp.predict(X), (-1,))

test_npy = np.reshape(sgp_plain.predict(test_X), (-1,))
test_py = np.reshape(sgp.predict(test_X, parametric_only=True), (-1,))
test_p2y = np.reshape(poly_model.predict(test_X), (-1,))
test_gy = np.reshape(sgp.predict(test_X), (-1,))

test_npK = sgp_plain.covariance(test_X, include_obs=True)
test_pK = sgp.covariance(test_X, parametric_only=True, include_obs=True)
test_p2K = poly_model.covariance(test_X, include_obs=True)
test_gK = sgp.covariance(test_X, include_obs=True)


realgp_K = k(X,X,True)
realgp_Kinv = np.linalg.inv(realgp_K)
realgp_alpha = np.dot(realgp_Kinv, y)
realgp_kstar = k(test_X, X, False)
test_rgp = np.dot(realgp_kstar, realgp_alpha)
test_rgpK = k(test_X, test_X, identical=True) - np.dot(realgp_kstar, np.dot(realgp_Kinv, realgp_kstar.T))

print "prediction errors"
print np.mean(np.abs(test_y))
print np.mean(np.abs(test_py-test_y))
print np.mean(np.abs(test_p2y-test_y))
print np.mean(np.abs(test_npy-test_y))
print np.mean(np.abs(test_gy-test_y))
print np.mean(np.abs(test_rgp-test_y))

def gaussian_KL(mean1, cov1, mean2, cov2):

    chol1 = np.linalg.cholesky(cov1)
    chol2 = np.linalg.cholesky(cov2)
    invcov2 = np.linalg.inv(cov2)

    k = len(mean1)
    mean_diff = mean2 - mean1
    qf = np.dot(mean_diff.T, np.dot(invcov2, mean_diff))

    tr = np.sum(invcov2 * cov1)

    logdet1 = np.log(np.diag(chol1)).sum()
    logdet2 = np.log(np.diag(chol2)).sum()

    kl = .5 * (tr + qf - k - logdet1 + logdet2)
    return kl

print "KLs"
print gaussian_KL(test_rgp, test_rgpK, test_py, test_pK)
print gaussian_KL(test_rgp, test_rgpK, test_p2y, test_p2K)
print gaussian_KL(test_rgp, test_rgpK, test_npy, test_npK)
print gaussian_KL(test_rgp, test_rgpK, test_gy, test_gK)
print gaussian_KL(test_rgp, test_rgpK, test_rgp, test_rgpK)

sys.exit(0)

savescatter(X, npy, "npy.png")
savescatter(X, gy, "gy.png")
savescatter(X, py, "py.png")
savescatter(X, p2y, "p2y.png")
savescatter(X, y, "y.png")

vmin = np.min(y)
vmax = np.max(y)
save_surface(lambda x,y: sgp.predict(np.array([[x, y],])), "surface_gy.png", vmin=vmin, vmax=vmax)
save_surface(lambda x,y: sgp.predict(np.array([[x, y],]), parametric_only=True), "surface_py.png", vmin=vmin, vmax=vmax)
save_surface(lambda x,y: np.dot(k(np.array([[x, y],]), X), realgp_alpha), "surface_rgpy.png", vmin=vmin, vmax=vmax)
save_surface(lambda x,y: sgp_plain.predict(np.array([[x, y],])), "surface_npy.png", vmin=vmin, vmax=vmax)
save_surface(lambda x,y: poly_model.predict(np.array([[x, y],])), "surface_p2y.png", vmin=vmin, vmax=vmax)

d = np.diag(test_rgpK)
vmin = np.min(d)
vmax = np.max(d)

#save_surface(lambda x,y: sgp.covariance(np.array([[x, y],])), "variance_gy.png", vmin=vmin, vmax=vmax)
#save_surface(lambda x,y: sgp.covariance(np.array([[x, y],]), parametric_only=True), "variance_py.png", vmin=vmin, vmax=vmax)
#save_surface(lambda x,y: k(np.array([[x, y],]), np.array([[x, y],])) - np.dot(k(np.array([[x, y],]), X), np.dot(realgp_Kinv, k(X, np.array([[x, y],])))), "variance_rgpy.png", vmin=vmin, vmax=vmax)
#save_surface(lambda x,y: sgp_plain.covariance(np.array([[x, y],])), "variance_npy.png", vmin=vmin, vmax=vmax)
#save_surface(lambda x,y: poly_model.covariance(np.array([[x, y],])), "variance_p2y.png", vmin=vmin, vmax=vmax)
