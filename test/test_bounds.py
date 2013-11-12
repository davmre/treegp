import numpy as np
import scipy.sparse
import pyublas
from sparsegp.cover_tree import VectorTree, MatrixTree


dfn_str = "euclidean"
dfn_params = np.array([1.0,1.0, 1.0], dtype=float)

wfn_str = "compact2"
wfn_params = np.array([2.0,], dtype=float)

tree_X = np.array([[0.0,0.0, 0.0],], dtype=float)
Kinv = scipy.sparse.eye(1)
nzr, nzc = Kinv.nonzero()
double_tree = MatrixTree(tree_X, nzr, nzc, dfn_str, dfn_params, wfn_str, wfn_params)

double_tree.test_bounds(2.5, 1000)
