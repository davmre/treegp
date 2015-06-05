


import sys
import os
import numpy as np
from treegp.bcm.bcmopt import SampledData, load_log
import  cPickle as pickle


d = sys.argv[1]
if d.endswith("/"):
    d = d[:-1]

segs = os.path.basename(d).split("_")
ntrain = int(segs[0])
n = int(segs[1])
lscale = float(segs[3])
obs_std = float(segs[4])
yd = 50
seed = int(segs[-1][1:])

sdf = "%d_%d_%.2f_%.3f_%d_%d.pkl" % (n, ntrain, lscale, obs_std, yd, seed)
sdf = os.path.join("bcmopt_experiments/synthetic_datasets/", sdf)
with open(sdf, 'rb') as f:
    sdata = pickle.load(f)

steps, times, lls = load_log(d)
best_idx = np.argmax(lls)
step = steps[best_idx]

rfname = os.path.join(d, "fast_results.txt")
results = open(rfname, 'w')

fname_X = os.path.join(d, "step_%05d_X.npy" % step)
X = np.load(fname_X)
l1 = sdata.mean_abs_err(X.flatten())
l2 = sdata.median_abs_err(X.flatten())

print d
print "mean error", l1
print "mean error", l2
