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

def eval_gp(dataset_name, model_name, tag=None, sgp=None, n=None, test_n=None, cutoff_rule=1):
    X_test, y_test = test_data(dataset_name)
    gp = trained_gp(dataset_name, model_name, n=n, tag=tag, build_tree=True)
    print "loaded GP, evaluating timings on %d test points..." % test_n

    resultfile = timing_results_fname(dataset_name, model_name, tag)
    errorfile = timing_errors_fname(dataset_name, model_name, tag)

    if test_n is None:
        test_n = len(X_test)

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
        naive_predict[i] = gp.predict_naive(X_test[i:i+1,:])
        t1 = time.time()
        naive_predict_times[i] = t1-t0


    for i in range(test_n):
        t2 = time.time()
        tree_predict[i] = gp.predict(X_test[i:i+1,:], eps=1e-4)
        t3 = time.time()
        tree_predict_times[i] = t3-t2
        tree_predict_terms[i] = gp.predict_tree.fcalls


    for i in range(test_n):
        t4 = time.time()
        sparse_covar[i] = gp.covariance(X_test[i:i+1,:])
        t5 = time.time()
        sparse_covar_times[i] = t5-t4
        sparse_covar_qftimes[i] = gp.qf_time
        sparse_covar_nonqftimes[i] = gp.nonqf_time



    for i in range(test_n):
        t41 = time.time()
        sparse_covar_spkernel[i] = gp.covariance_spkernel(X_test[i:i+1,:])
        t51 = time.time()
        sparse_covar_spkernel_times[i] = t51-t41
        sparse_covar_spkernel_qftimes[i] = gp.qf_time
        sparse_covar_spkernel_nonqftimes[i] = gp.nonqf_time

    for i in range(test_n):
        t41 = time.time()
        sparse_covar_spkernel_solve[i] = gp.covariance_spkernel_solve(X_test[i:i+1,:])
        t51 = time.time()
        sparse_covar_spkernel_solve_times[i] = t51-t41


    has_dense = True
    try:
        for i in range(100):
            t42 = time.time()
            dense_covar[i] = gp.covariance_dense(X_test[i:i+1,:])
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
                tree_covar[e_i, e_j, i] = gp.covariance_double_tree(X_test[i:i+1,:], eps=10**(-epsm), eps_abs=10**(-eps_abs), cutoff_rule=cutoff_rule)
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

    print "wrote results to", resultfile

def main():

    # inputs
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    if len(sys.argv) > 3:
        tag = sys.argv[3]
    else:
        tag = None

    eval_gp(dataset_name, model_name, test_n=100, tag=tag)

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
