import os, errno
import sys
import numpy as np
import scipy.linalg
import matplotlib
import itertools
import time
from scikits.sparse.cholmod import cholesky as sp_cholesky
import scipy.sparse
import scipy.stats

from datasets import *
from prediction import trained_gp

def strstats(v):
    if v.dtype == float:
        return "mean %f std %f min %f 10th %f 50th %f 90th %f max %f" % stats(v)
    elif v.dtype == int:
        return "mean %d std %d min %d 10th %d 50th %d 90th %d max %d" % stats(v)
    else:
        raise Exception("weird dtype %s"% v.dtype)

def stats(v):
  return (np.mean(v), np.std(v), np.min(v), scipy.stats.scoreatpercentile(v, 10), np.median(v) ,scipy.stats.scoreatpercentile(v, 90) , np.max(v))


def profile_tree(dataset_name, model_name, tag=None, sgp=None, n=None, test_n=None, burnin=10, cutoff_rule=2):

    X_test, y_test = test_data(dataset_name)
    gp = trained_gp(dataset_name, model_name, n=n, tag=tag, build_tree=True, leaf_bin_width=0.5, build_dense_Kinv_hack=True)

    if test_n is None:
        test_n = min(len(X_test), 5000)
    print "loaded GP, evaluating timings on %d test points..." % test_n

    eps_abs = 0.001 * gp.noise_var

    tree_covar = np.zeros((test_n,))
    tree_covar_terms = np.zeros((test_n,))
    tree_covar_distevals = np.zeros((test_n,))
    tree_covar_times = np.zeros((test_n,))
    tree_covar_qftimes = np.zeros((test_n,))
    tree_covar_nonqftimes = np.zeros((test_n,))



    gp.nonqf_time = 0
    for i in range(test_n):
        t6 = time.time()
        tree_covar[i] = gp.covariance_double_tree(X_test[i:i+1,:], eps_abs=eps_abs, cutoff_rule=cutoff_rule)
        t7 = time.time()

        tree_covar_terms[i] = gp.double_tree.fcalls
        tree_covar_distevals[i] = gp.double_tree.dfn_evals
        tree_covar_times[i] = t7-t6
        tree_covar_qftimes[i] = gp.qf_time
        tree_covar_nonqftimes[i] = gp.nonqf_time

    print "tree: eps_abs %f times: %s" % (eps_abs, strstats(tree_covar_times[burnin:]))
    print "tree: eps_abs %f qftimes: %s" % (eps_abs, strstats(tree_covar_qftimes[burnin:]))
    print "tree: eps_abs %f nqftimes: %s" % (eps_abs, strstats(tree_covar_nonqftimes[burnin:]))
    print "tree: eps_abs %f terms: %s" % (eps_abs, strstats(tree_covar_terms[burnin:]))
    print "tree: eps_abs %f dfnevals: %s" % (eps_abs, strstats(tree_covar_distevals[burnin:]))



def eval_gp(dataset_name, model_name, tag=None, sgp=None, n=None, test_n=None, burnin=10, cutoff_rule=2):
    X_test, y_test = test_data(dataset_name)
    gp = trained_gp(dataset_name, model_name, n=n, tag=tag, build_tree=True, leaf_bin_width=None, build_dense_Kinv_hack=True, compile_tree=compiled_tree_fname(dataset_name, model_name, tag))
    if test_n is None:
        test_n = min(len(X_test), 5000)


    print "loaded GP, evaluating timings on %d test points..." % test_n

    resultfile = timing_results_fname(dataset_name, model_name, tag)
    errorfile = timing_errors_fname(dataset_name, model_name, tag)




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

    sparse_covar_local = np.zeros(test_n)
    sparse_covar_local_times = np.zeros(test_n)
    sparse_covar_local_tree_times = np.zeros(test_n)
    sparse_covar_local_math_times = np.zeros(test_n)
    sparse_covar_local_terms = np.zeros(test_n)
    sparse_covar_local_dfn_evals = np.zeros(test_n)
    sparse_covar_local_wfn_evals = np.zeros(test_n)
    sparse_covar_local_qftimes = np.zeros(test_n)
    sparse_covar_local_nonqftimes = np.zeros(test_n)


    sparse_covar_spkernel_solve = np.zeros(test_n)
    sparse_covar_spkernel_solve_times = np.zeros(test_n)

    print "naive predict"
    for i in range(test_n):
        t0 = time.time()
        naive_predict[i] = gp.predict_naive(X_test[i:i+1,:])
        t1 = time.time()
        naive_predict_times[i] = t1-t0

    print "tree predict"
    for i in range(test_n):
        t2 = time.time()
        tree_predict[i] = gp.predict(X_test[i:i+1,:], eps=1e-4)
        t3 = time.time()
        tree_predict_times[i] = t3-t2
        tree_predict_terms[i] = gp.predict_tree.terms

    print "treedense/local covar"
    gp.nonqf_time = 0
    for i in range(test_n):
        t4 = time.time()
        sparse_covar_local[i] = gp.covariance_treedense(X_test[i:i+1,:])
        t5 = time.time()
        sparse_covar_local_times[i] = t5-t4
        sparse_covar_local_tree_times[i] = gp.predict_tree.dense_hack_tree_s
        sparse_covar_local_math_times[i] = gp.predict_tree.dense_hack_math_s
        sparse_covar_local_terms[i] = gp.qf_terms
        sparse_covar_local_dfn_evals[i] = gp.qf_dfn_evals
        sparse_covar_local_wfn_evals[i] = gp.qf_wfn_evals
        sparse_covar_local_qftimes[i] = gp.qf_time
        sparse_covar_local_nonqftimes[i] = gp.nonqf_time

    print "sparse covar"
    gp.nonqf_time = 0
    for i in range(test_n):
        t4 = time.time()
        sparse_covar[i] = gp.covariance(X_test[i:i+1,:])
        t5 = time.time()
        sparse_covar_times[i] = t5-t4
        sparse_covar_qftimes[i] = gp.qf_time
        sparse_covar_nonqftimes[i] = gp.nonqf_time



    print "spkernel"
    for i in range(test_n):
        t41 = time.time()
        sparse_covar_spkernel[i] = gp.covariance_spkernel(X_test[i:i+1,:])
        t51 = time.time()
        sparse_covar_spkernel_times[i] = t51-t41
        sparse_covar_spkernel_qftimes[i] = gp.qf_time
        sparse_covar_spkernel_nonqftimes[i] = gp.nonqf_time

    """
    for i in range(test_n):
        t41 = time.time()
        sparse_covar_spkernel_solve[i] = gp.covariance_spkernel_solve(X_test[i:i+1,:])
        t51 = time.time()
        sparse_covar_spkernel_solve_times[i] = t51-t41
    """

    has_dense = True
    try:
        for i in range(100):
            t42 = time.time()
            dense_covar[i] = gp.covariance_dense(X_test[i:i+1,:])
            t52 = time.time()
            dense_covar_times[i] = t52-t42
    except:
        has_dense=False

    f = open(resultfile, 'w')

    eps_abs = 0.001 * gp.noise_var

    tree_covar = np.zeros((test_n,))
    tree_covar_terms = np.zeros((test_n,))
    tree_covar_zeroterms = np.zeros((test_n,))
    tree_covar_nodes_touched = np.zeros((test_n,))
    tree_covar_dfn_evals = np.zeros((test_n,))
    tree_covar_wfn_evals = np.zeros((test_n,))
    tree_covar_dfn_misses = np.zeros((test_n,))
    tree_covar_wfn_misses = np.zeros((test_n,))
    tree_covar_times = np.zeros((test_n,))
    tree_covar_qftimes = np.zeros((test_n,))
    tree_covar_nonqftimes = np.zeros((test_n,))
    for i in range(test_n):
        t6 = time.time()
        tree_covar[i] = gp.covariance_double_tree(X_test[i:i+1,:], eps_abs=eps_abs, cutoff_rule=cutoff_rule)
        t7 = time.time()

        tree_covar_terms[i] = gp.qf_terms
        tree_covar_zeroterms[i] = gp.qf_zeroterms
        tree_covar_nodes_touched[i] = gp.qf_nodes_touched
        tree_covar_dfn_evals[i] = gp.qf_dfn_evals
        tree_covar_wfn_evals[i] = gp.qf_wfn_evals
        tree_covar_dfn_misses[i] = gp.qf_dfn_misses
        tree_covar_wfn_misses[i] = gp.qf_wfn_misses


        tree_covar_times[i] = t7-t6
        tree_covar_qftimes[i] = gp.qf_time
        tree_covar_nonqftimes[i] = gp.nonqf_time

    tree_compiled = np.zeros((test_n,))
    tree_compiled_terms = np.zeros((test_n,))
    tree_compiled_zeroterms = np.zeros((test_n,))
    tree_compiled_nodes_touched = np.zeros((test_n,))
    tree_compiled_dfn_evals = np.zeros((test_n,))
    tree_compiled_times = np.zeros((test_n,))
    tree_compiled_qftimes = np.zeros((test_n,))
    tree_compiled_nonqftimes = np.zeros((test_n,))
    for i in range(test_n):
        t6 = time.time()
        tree_compiled[i] = gp.covariance_compiled(X_test[i:i+1,:], eps_abs=eps_abs)
        t7 = time.time()

        tree_compiled_terms[i] = gp.qf_terms
        tree_compiled_zeroterms[i] = gp.qf_zeroterms
        tree_compiled_nodes_touched[i] = gp.qf_nodes_touched
        tree_compiled_dfn_evals[i] = gp.qf_dfn_evals

        tree_compiled_times[i] = t7-t6
        tree_compiled_qftimes[i] = gp.qf_time
        tree_compiled_nonqftimes[i] = gp.nonqf_time


    f.write("naive predict times: %s\n" % strstats(naive_predict_times[burnin:]))
    f.write("\n")
    f.write("tree predict times: %s\n" % strstats(tree_predict_times[burnin:]))
    f.write("tree predict terms:  %s\n" % strstats(tree_predict_terms[burnin:]))
    f.write("tree predict errors: %s\n" % strstats(np.abs(tree_predict[burnin:] - naive_predict[burnin:])))
    f.write("\n")

    if has_dense:
        f.write("dense covar times: %s\n" % strstats(dense_covar_times[burnin:]))
    f.write("sparse covar times: %s\n" % strstats(sparse_covar_times[burnin:]))
    f.write("sparse covar qftimes: %s\n" % strstats(sparse_covar_qftimes[burnin:]))
    f.write("sparse covar nqftimes: %s\n" % strstats(sparse_covar_nonqftimes[burnin:]))
    f.write("\n")
    f.write("sparse covar spkernel times: %s\n" % strstats(sparse_covar_spkernel_times[burnin:]))
    f.write("sparse covar spkernel qftimes: %s\n" % strstats(sparse_covar_spkernel_qftimes[burnin:]))
    f.write("sparse covar spkernel nqftimes: %s\n" % strstats(sparse_covar_spkernel_nonqftimes[burnin:]))
    f.write("sparse covar spkernel error: %f\n" % np.mean(np.abs(sparse_covar_spkernel[burnin:] - sparse_covar[burnin:])))
    f.write("\n")

    f.write("sparse covar local times: %s\n" % strstats(sparse_covar_local_times[burnin:]))
    f.write("sparse covar local tree time: %s\n" % strstats(sparse_covar_local_tree_times[burnin:]))
    f.write("sparse covar local math time: %s\n" % strstats(sparse_covar_local_math_times[burnin:]))
    f.write("sparse covar local tree/math time ratio: %s\n" % strstats(sparse_covar_local_tree_times[burnin:]/sparse_covar_local_math_times[burnin:]))
    f.write("sparse covar local terms: %s\n" % strstats(sparse_covar_local_terms[burnin:]))
    f.write("sparse covar local dfns: %s\n" % strstats(sparse_covar_local_dfn_evals[burnin:]))
    f.write("sparse covar local wfns: %s\n" % strstats(sparse_covar_local_wfn_evals[burnin:]))
    f.write("sparse covar local qftimes: %s\n" % strstats(sparse_covar_local_qftimes[burnin:]))
    f.write("sparse covar local nqftimes: %s\n" % strstats(sparse_covar_local_nonqftimes[burnin:]))
    f.write("sparse covar local error: %s\n" % strstats(np.abs(sparse_covar_local[burnin:] - sparse_covar[burnin:])))
    f.write("\n")


    """
    f.write("sparse covar spkernel_solve times: %s\n" % strstats(sparse_covar_spkernel_solve_times))
    f.write("sparse covar spkernel_solve error: %f\n" % np.mean(np.abs(sparse_covar_spkernel_solve - sparse_covar)))
    f.write("\n")
    """

    f.write("actual vars: %s\n" % strstats(sparse_covar[burnin:]))
    f.write("actual qfs: %s\n" % strstats(1-sparse_covar[burnin:]))
    f.write("\n")

    f.write("tree: eps_abs %f times: %s\n" % (eps_abs, strstats(tree_covar_times[burnin:])))
    f.write("tree: eps_abs %f terms: %s\n" % (eps_abs, strstats(tree_covar_terms[burnin:])))
    f.write("tree: eps_abs %f zeroterms: %s\n" % (eps_abs, strstats(tree_covar_zeroterms[burnin:])))
    f.write("tree: eps_abs %f nodes touched: %s\n" % (eps_abs, strstats(tree_covar_nodes_touched[burnin:])))
    f.write("tree: eps_abs %f dfn calls: %s\n" % (eps_abs, strstats(tree_covar_dfn_evals[burnin:])))
    f.write("tree: eps_abs %f wfn calls: %s\n" % (eps_abs, strstats(tree_covar_wfn_evals[burnin:])))
    f.write("tree: eps_abs %f dfn cache misses: %s\n" % (eps_abs, strstats(tree_covar_dfn_misses[burnin:])))
    f.write("tree: eps_abs %f wfn cache misses: %s\n" % (eps_abs, strstats(tree_covar_wfn_misses[burnin:])))

    f.write("tree: eps_abs %f qftimes: %s\n" % (eps_abs, strstats(tree_covar_qftimes[burnin:])))
    f.write("tree: eps_abs %f nqftimes: %s\n" % (eps_abs, strstats(tree_covar_nonqftimes[burnin:])))
    f.write("tree: eps_abs %f rel errors: %s \n" %  (eps_abs, strstats(np.abs((tree_covar[burnin:] - sparse_covar[burnin:])/(1-sparse_covar[burnin:])))))
    f.write("tree: eps_abs %f var-rel errors: %s \n" %  (eps_abs, strstats(np.abs((tree_covar[burnin:] - sparse_covar[burnin:])/sparse_covar[burnin:]))))
    f.write("tree: eps_abs %f abs errors: %s \n" %  (eps_abs, strstats(np.abs(tree_covar[burnin:] - sparse_covar[burnin:]))))
    f.write("\n")

    f.write("compiled tree: eps_abs %f times: %s\n" % (eps_abs, strstats(tree_compiled_times[burnin:])))
    f.write("compiled tree: eps_abs %f terms: %s\n" % (eps_abs, strstats(tree_compiled_terms[burnin:])))
    f.write("compiled tree: eps_abs %f zeroterms: %s\n" % (eps_abs, strstats(tree_compiled_zeroterms[burnin:])))
    f.write("compiled tree: eps_abs %f nodes touched: %s\n" % (eps_abs, strstats(tree_compiled_nodes_touched[burnin:])))
    f.write("compiled tree: eps_abs %f dfn calls: %s\n" % (eps_abs, strstats(tree_compiled_dfn_evals[burnin:])))
    f.write("compiled tree: eps_abs %f qftimes: %s\n" % (eps_abs, strstats(tree_compiled_qftimes[burnin:])))
    f.write("compiled tree: eps_abs %f nqftimes: %s\n" % (eps_abs, strstats(tree_compiled_nonqftimes[burnin:])))
    f.write("compiled tree: eps_abs %f rel errors: %s \n" %  (eps_abs, strstats(np.abs((tree_compiled[burnin:] - sparse_covar[burnin:])/(1-sparse_covar[burnin:])))))
    f.write("compiled tree: eps_abs %f var-rel errors: %s \n" %  (eps_abs, strstats(np.abs((tree_compiled[burnin:] - sparse_covar[burnin:])/sparse_covar[burnin:]))))
    f.write("compiled tree: eps_abs %f abs errors: %s \n" %  (eps_abs, strstats(np.abs(tree_compiled[burnin:] - sparse_covar[burnin:]))))
    f.write("\n")


    max_i = np.argmax(tree_covar_times[burnin:]) + burnin
    f.write("most expensive variance point was %s at cost %f\n" % (X_test[max_i,:], tree_covar_times[max_i]))

    f.close()

    np.savez(errorfile, tree_compiled=tree_compiled, tree_covar=tree_covar, sparse_covar=sparse_covar, tree_predict=tree_predict, naive_predict=naive_predict, sparse_covar_spkernel=sparse_covar_spkernel, burnin=burnin)

    print "wrote results to", resultfile

def main():

    # inputs
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    if len(sys.argv) > 3:
        tag = sys.argv[3]
    else:
        tag = None

    #profile_tree(dataset_name, model_name, tag=tag, test_n=700)
    eval_gp(dataset_name, model_name, tag=tag, test_n=None, burnin=5)

    #print "timings finished"

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import sys, traceback, pdb
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
