from sparsegp.gp import GPCov
import sys
import os
import cPickle as pickle
import numpy as np

outfile = "hyperparams_matlab.pkl"

if os.path.exists('dfn_params_fic.txt'):
    # convert CSFIC hyperparams

    dfn_fic = np.reshape(np.loadtxt("dfn_params_fic.txt", delimiter=','), (-1,))
    wfn_fic = np.reshape(np.loadtxt("wfn_params_fic.txt", delimiter=','), (-1,))
    dfn_cs = np.reshape(np.loadtxt("dfn_params_cs.txt", delimiter=','), (-1,))
    wfn_cs = np.reshape(np.loadtxt("wfn_params_cs.txt", delimiter=','), (-1,))
    dfn_fic *= np.sqrt(2.0)

    Xu = np.loadtxt("Xu.txt", delimiter=',')
    noise_var = np.loadtxt('noise_var.txt')

    cov_main = GPCov(dfn_str="euclidean", dfn_params=dfn_cs, wfn_str="compact2", wfn_params=wfn_cs)
    cov_fic = GPCov(dfn_str="euclidean", dfn_params=dfn_fic, wfn_str="se", wfn_params=wfn_fic, Xu=Xu)

    hparams = dict()
    hparams['cov_main']=cov_main
    hparams['cov_fic']=cov_fic
    hparams['noise_var']=noise_var


    if len(sys.argv) > 1:
        outfile = os.path.join("/home/dmoore/python/sparsegp/experiments/models/%s/csfic%d/" % (sys.argv[1],Xu.shape[0]), outfile)

    with open(outfile, 'w') as f:
        pickle.dump(hparams, f)

else:
    # convert SE hyperparams

    dfn = np.reshape(np.loadtxt("dfn_params.txt", delimiter=','), (-1,))
    wfn = np.reshape(np.loadtxt("wfn_params.txt", delimiter=','), (-1,))
    noise_var = np.loadtxt('noise_var.txt')
    dfn *= np.sqrt(2.0)
    cov_main = GPCov(dfn_str="euclidean", dfn_params=dfn, wfn_str="se", wfn_params=wfn)

    hparams = dict()
    hparams['cov_main']=cov_main
    hparams['cov_fic']=None
    hparams['noise_var']=noise_var

    if len(sys.argv) > 1:
        outfile = os.path.join("/home/dmoore/python/sparsegp/experiments/models/%s/se/" % sys.argv[1], outfile)

    with open(outfile, 'w') as f:
        pickle.dump(hparams, f)

print "dumped to", outfile
