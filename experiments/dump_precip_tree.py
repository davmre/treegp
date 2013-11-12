import pandas as pd
from sigvisa.utils.cover_tree import VectorTree
import numpy as np
import pyublas

X = pd.read_csv('datasets/precip/precip_all_X.txt')
X['elev'] = 0
Xlld = np.array(X, copy=True)
t = VectorTree(Xlld, 1, 'lld', np.array([1.0, 1.0]), 'se', np.array([1.0,]))
t.dump_tree('preciptree.txt')
