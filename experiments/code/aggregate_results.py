import numpy as np
import re
from sparsegp.experiments.code.datasets import predict_results_fname, timing_results_fname, gp_fname
from sparsegp.gp import GP
import scipy.sparse
model_list_fname = "models"

basedir = "experiments/models/"

class NoResultsError(Exception):
    pass

def extract_results(txt, prefix, require_times=True):
    for line in txt:
        if line.startswith(prefix):
            if (not require_times) or "times" in line:
                d = dict()
                for w in ("mean", "std", "min", "10th", "50th", "90th", "max"):
                    d[w] = float(re.search(r"%s ([\.\d]+)" % w, line).group(1))
                return d

    raise NoResultsError("could not find line with prefix %s" % prefix)

def parse_timings(timings_lines):

    sparse = extract_results(timings_lines, "sparse covar")
    hybrid = extract_results(timings_lines, "sparse covar spkernel")
    tree = extract_results(timings_lines, "tree: eps_abs")

    return sparse, hybrid, tree


with open(model_list_fname, 'r') as f:
    model_lines = f.readlines()

print_timings = False

for line in model_lines:
    dataset, model, tag = line.strip().split()

    accuracy_fname = predict_results_fname(dataset, model, tag)
    timing_fname = timing_results_fname(dataset, model, tag)
    trained_fname = gp_fname(dataset, model, tag=tag)

    if print_timings:
        try:
            with open(timing_fname, 'r') as f:
                timings_lines = f.readlines()
        except IOError:
            continue
        sparse, hybrid, tree = parse_timings(timings_lines)
        print dataset, model, sparse['mean']*1000, hybrid['mean']*1000, tree['mean']*1000
    else:
        with open(accuracy_fname, 'r') as f:
            acc_lines = f.readlines()
        msll = float(acc_lines[0].split()[1])
        smse = float(acc_lines[1].split()[1])

        sgp = GP(fname=trained_fname, build_tree=False)

        if scipy.sparse.issparse(sgp.Kinv):
            fullness = float(len(sgp.Kinv.nonzero()[0])) / sgp.Kinv.shape[0]**2
        else:
            fullness = float(np.sum(np.abs(sgp.Kinv) > sgp.sparse_threshold))  / sgp.Kinv.shape[0]**2
        fullness *= 100.0

        print dataset, model, fullness, msll, smse
