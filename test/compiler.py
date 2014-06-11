import numpy as np
import pyublas
import types
import os

from treegp.gp import GP, GPCov

def get_trivial_Xy():
    X = np.array( ((0.0, 0.0), (0.1, -0.1), (1.0, 1.0), (1.0, 0.0) ) )
    y = np.array((0.1, 0.2, -3, -1))
    return X, y

def build_gp(compile_tree=None):
    X, y = get_trivial_Xy()

    cov = GPCov(wfn_params=[2.0,], dfn_params=[ 0.2, 0.3], wfn_str="compact2", dfn_str="euclidean")
    noise_var = 0.01
    gp = GP(X=X, y=y, noise_var=noise_var, cov_main=cov, compute_ll=False, leaf_bin_width=0.5, build_tree=True, compile_tree=compile_tree)
    return gp

def test_compiler():
    gp = build_gp()
    print "compiling"
    gp.double_tree.compile("compiled_tree.cc", 1)
    print "done"

    #os.system("gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC -I/home/dmoore/.virtualenvs/treegp/local/lib/python2.7/site-packages/pyublas/include -I/home/dmoore/.virtualenvs/treegp/local/lib/python2.7/site-packages/numpy/core/include -I/home/dmoore/local/include/ -I/usr/include/python2.7 -c compiled_tree.cc -o compiled_tree.o -g -O0")
    #os.system("g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro compiled_tree.o -L/ -lboost_python -o /home/dmoore/python/treegp/compiled_tree.so")



def test_compiled_tree():
    import compiled_tree
    compiled_tree.init_distance_caches()
    testX = np.array( ((0.05, 0.05),))
    r = compiled_tree.quadratic_form_symmetric(testX, 1e-8)
    print "got qf", r

    print "debug terms", compiled_tree.get_terms()
    print "debug zeroterms", compiled_tree.get_zeroterms()
    print "debug nodes touched", compiled_tree.get_nodes_touched()
    print "debug dfn evals", compiled_tree.get_dfn_evals()

    gp = build_gp()
    r2 = gp.double_tree.quadratic_form(testX, testX, 1e-8, 1e-8, 2)
    print "got interpreted qf", r2

def test_auto_compiled_tree():
    testX = np.array( ((0.05, 0.05),))
    gp = build_gp(compile_tree="autocompile")
    r1 = gp.compiled_tree.quadratic_form_symmetric(testX, 1e-8)
    r2 = gp.double_tree.quadratic_form(testX, testX, 1e-8, 1e-8, 2)
    print "got qfs", r1, r2

    c1 = gp.covariance_compiled(testX)
    c2 = gp.covariance_double_tree(testX)
    print "got covs", c1, c2

if __name__ == "__main__":
    test_auto_compiled_tree()
