from sigvisa.models.spatial_regression.SparseGP import SparseGP
from sigvisa.utils.cover_tree import VectorTree
import pyublas
import scipy.linalg
import numpy as np


def savescatter(fname, x, y):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    f = Figure()
    ax = f.add_subplot(111)
    ax.scatter(x, y, c='black')
    ax.set_xlim(0, 1)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname)

def saveplot(x, y, fname):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    permute = x.argsort()
    x = x[permute]
    y = y[permute]

    f = Figure()
    ax = f.add_subplot(111)
    ax.plot(x, y, c='black')
    ax.set_xlim(0, 1)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname)


n = 500
test_n = 500

X = np.random.rand(n+test_n,1)

long_scale=1.0
short_scale=.02
sigma2_n = 0.1
sigma2_f_long = 100.0
sigma2_f_short = 1.0

dfn_params_long=np.array([long_scale, ], dtype=float)
dfn_params_short=np.array([short_scale,], dtype=float)
wfn_params_long = np.array([sigma2_f_long,], dtype=float)
wfn_params_short = np.array([sigma2_f_short,], dtype=float)
long_tree = VectorTree(X[0:1,:], 1, "euclidean", dfn_params_long, "se", wfn_params_long)
long_kernel = long_tree.kernel_matrix(X, X, False)
short_tree = VectorTree(X[0:1,:], 1, "euclidean", dfn_params_short, "se", wfn_params_short)
short_kernel = short_tree.kernel_matrix(X, X, False)

kernel = long_kernel + short_kernel + sigma2_n * np.eye(n+test_n)
L = scipy.linalg.cholesky(kernel, lower=True)
z = np.random.randn(n+test_n)
y = np.dot(L, z)

test_X = X[n:, :]
X = X[:n,:]
test_y = y[n:]
y = y[:n]


def poly_basisfns_1d(order):
    basisfn_strs = ["lambda x : " + ("x[0]**%d " % (d1,)) for d1 in range(order+1)]
    return [eval(s) for s in basisfn_strs]
bfn = poly_basisfns_1d(5)
d = len(bfn)
b = np.zeros((d,))
B = np.eye(d) * 10
print bfn

hyperparams = [sigma2_n, sigma2_f_short, short_scale, short_scale]
sgp  = SparseGP(X=X, y=y, basisfns = bfn, param_mean=b, param_cov=B, dfn_str="euclidean", wfn_str="se", hyperparams=hyperparams, sort_events=False, build_tree=False)

sgp_plain  = SparseGP(X=X, y=y, basisfns = [], param_mean=None, param_cov=None, dfn_str="euclidean", wfn_str="se", hyperparams=hyperparams, sort_events=False, build_tree=False)

test_npy = np.reshape(sgp_plain.predict_naive(test_X), (-1,))
npy = np.reshape(sgp_plain.predict_naive(X), (-1,))

test_py = np.reshape(sgp.predict_naive(test_X, parametric_only=True), (-1,))
py = np.reshape(sgp.predict_naive(X, parametric_only=True), (-1,))
gy =  np.reshape(sgp.predict_naive(X), (-1,))

test_gy = np.reshape(sgp.predict_naive(test_X), (-1,))

print np.median(np.abs(test_y))
print np.median(np.abs(test_py-test_y))
print np.median(np.abs(test_npy-test_y))
print np.median(np.abs(test_gy-test_y))

saveplot(X[:,0], npy, "npy.png")
saveplot(X[:,0], gy, "gy.png")
saveplot(X[:,0], py, "py.png")
saveplot(X[:,0], y, "y.png")
