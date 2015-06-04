from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker
import numpy as np
from treegp.gp import GP, GPCov, mcov, dgaussian, dgaussian_rank1
import time

X = np.random.rand(10000, 2)
Y = np.random.randn(10000, 50)
lscale=0.4
noise_var=0.01
block_boundaries = [(0, 10000)]
cov = GPCov(wfn_params=[1.0], dfn_params=[lscale, lscale], dfn_str="euclidean", wfn_str="se")
mbcm = MultiSharedBCM(X, Y, block_boundaries, cov, noise_var, neighbor_threshold=1e-3)

for n  in range(200, 2000, 200):
    XX = X[:n]
    YY = Y[:n]
    t0  =time.time()
    mbcm.gaussian_llgrad(XX, YY, grad_X = True)
    t1 = time.time()
    print "llgrad %d %.3f" % (n, t1-t0)

for n  in range(200, 2000, 200):
    XX = X[:n]
    YY = Y[:n]
    t0  =time.time()
    mbcm.gaussian_llgrad(XX, YY, grad_X = False)
    t1 = time.time()
    print "ll %d %.3f" % (n, t1-t0)
