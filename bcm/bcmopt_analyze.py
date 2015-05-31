from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic
from treegp.bcm.local_regression import BCM
from treegp.bcm.bcmopt import SampledData

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from treegp.util import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import time
import os
import sys

import cPickle as pickle

from matplotlib.figure import Figure

RESULT_COLS = {'step': 0, 'time': 1, 'mll': 2, 'mad': 3,
               'xprior': 4, 'predll': 5, 'predll_neighbors': 6}

def plot_ll(run_name):
    steps, times, lls = load_log(run_name)

def load_results(d):
    r = os.path.join(d, "results.txt")

    results = []
    with open(r, 'r') as rf:
        for line in rf:
            try:
                lr = [float(x) for x in line.split(' ')]
                results.append(lr)
            except:
                continue
    return np.asarray(results)

def read_result_line(s):
    r = {}
    parts = s.split(' ')
    for lbl, col in RESULT_COLS.items():
        p = parts[col]
        if p=="trueX": continue
        intP = int(p)
        floatP = float(p)
        if float(intP) == floatP:
            r[lbl] = intP
        else:
            r[lbl] = floatP
    return r

def load_final_results(d):
    r = os.path.join(d, "results.txt")

    results = []
    with open(r, 'r') as rf:
        lines = rf.readlines()
        r_final = read_result_line(lines[-2])
        r_true = read_result_line(lines[-1])
    return r_final, r_true

def load_plot_data(runs, target="predll"):

    col = RESULT_COLS[target]

    plot_data = {}
    for label, run_params in runs.items():
        results = load_results(exp_dir(run_params))
        t = results[:, 1]
        y = results[:, col]
        plot_data[label] = (t, y)

    return plot_data

def write_plot(plot_data, out_fname, xlabel="Time (s)", ylabel)

    fig = Figure(dpi=144)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)

    for label, (x, y) in plot_data.items():
        ax.plot(x, y, label=label)

    ax.legend()
    fig.savefig(out_fname)


def plot_models_fixedsize():
    ntrain = 15000
    n = 15550
    lscale = 0.4
    obs_std = 0.1
    yd = yd
    seed=4
    local_dist=0.05
    method="l-bfgs-b"

    base_params = {'ntrain': ntrain, 'n': n, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': local_dist, "method": method, 'nblocks': 1}

    runs = {'GP': base_params}
    block_counts = [4, 9, 16, 25, 36, 39]
    for bc in block_counts:
        rfp = base_params.copy()
        rfp['nblocks'] = bc
        runs['GPRF-%d' % bc] = rfp

        localp = rfp.copy()
        localp['local_dist'] = 0.0
        runs['Local-%d' % d] = localp

    for target in ("predll", "predll_neighbors", "mad"):
        plot_data = load_plot_data(runs, target=target)
        write_plot(plot_data, out_fname="fixedsize_%s.png" % target, ylabel=target)



def plot_models_growing():

    yd = 50
    seed = 4
    method = "l-bfgs-b"
    ntest  =500

    ntrains = [500, 2000, 4500, 8000, 12500, 18000, 24500]
    nblocks = [1, 4, 9, 16, 25, 36, 49]
    lscales = [1.0, 0.66666, 0.5, 0.4, 0.333333, 0.2856]
    obs_stds = [0.2, 0.1333333, 0.1, 0.08, 0.0666666, 0.05714]

    x = ntrains
    runs_gprf = []
    runs_local = []
    runs_full = []

    for ntrain, nblock, lscale, obs_std in zip(ntrains, nblocks, lscales, obs_stds):
        run_params_gprf = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.05, "method": method, 'nblocks': nblock}
        runs_gprf.append(run_params_gprf)

        run_params_local = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.00, "method": method, 'nblocks': nblock}
        runs_local.append(run_params_local)

        run_params_full = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.00, "method": method, 'nblocks': 1}
        runs_full.append(run_params_full)

    times_local = []
    times_gprf = []
    times_full = []
    mad_local = []
    mad_gprf = []
    mad_full = []
    predll_local = []
    predll_gprf = []
    predll_full = []
    predll_true_local = []
    predll_true_gprf = []
    predll_true_full = []
    for (gprf, local, full) in zip(runs_gprf, runs_local, runs_full):

        def process(stuff, times, mads, predlls, predlls_true):
            d = exp_dir(stuff)
            r = load_results(d)
            times.append(np.mean(np.diff(r[:, 1])))
            fr, tr = load_final_results(d)
            mads.append(fr['mad'])
            predlls.append(fr['predll'])
            predlls_true.append(tr['predll'])

        process(gprf, times_gprf, mad_gprf, predll_gprf, predll_true_gprf)
        process(local, times_local, mad_local, predll_local, predll_true_local)
        process(full, times_full, mad_full, predll_full, predll_true_full)

    pd_times = {'GPRF': (ntrains, times_gprf),
                'Local': {ntrains, times_local},
                "GP": {trains, times_full}}
    pd_mad = {'GPRF': (ntrains, mad_gprf),
                'Local': {ntrains, mad_local},
                "GP": {trains, mad_full}}
    pd_predll = {'GPRF': (ntrains, predll_gprf),
                'Local': {ntrains, predll_local},
                "GP": {trains, predll_full}}
    pd_predll_true = {'GPRF': (ntrains, predll_true_gprf),
                'Local': {ntrains, predll_true_local},
                "GP": {trains, predll_true_full}}

    write_plot(pd_times, "times.png", xlabel="n", ylabel="gradient evaluation time (s)")
    write_plot(pd_mad, "mad.png", xlabel="n", ylabel="X locations: mean absolute deviation")
    write_plot(pd_predll, "predll.png", xlabel="n", ylabel="test MSLL")
    write_plot(pd_predll_true, "predll_true.png", xlabel="n", ylabel="test MSLL from true X")

def main():
    results_from_step(sys.argv[1], sys.argv[2])

if __name__ =="__main__":
    main()
