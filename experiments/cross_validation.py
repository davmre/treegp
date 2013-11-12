import numpy as np

import time
import hashlib
import os
import sys
import traceback
import pdb

class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def cv_generator(n, k=3):
    data = np.random.permutation(n)
    fold_size = n / k
    folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    folds[k - 1] = data[(k - 1) * fold_size:]
    for i in range(k):
        train = np.array(())
        for j in range(k):
            if j != i:
                train = np.concatenate([train, folds[j]])
        test = folds[i]
        yield ([int(t) for t in train], [int(t) for t in test])


def save_cv_folds(X, y, evids, cv_dir, folds=3):
    if os.path.exists(os.path.join(cv_dir, "fold_%02d_test.txt" % (folds - 1))):
        print "folds already exist, not regenerating."
        return

    np.savetxt(os.path.join(cv_dir, "X.txt"), X)
    np.savetxt(os.path.join(cv_dir, "y.txt"), y)
    np.savetxt(os.path.join(cv_dir, "evids.txt"), evids)

    for i, (train, test) in enumerate(cv_generator(len(y), k=folds)):
        np.savetxt(os.path.join(cv_dir, "fold_%02d_train.txt" % i), train)
        np.savetxt(os.path.join(cv_dir, "fold_%02d_test.txt" % i), test)


def train_cv_models(cv_dir, learn_model, model_type, **kwargs):
    X = np.loadtxt(os.path.join(cv_dir, "X.txt"))
    y = np.loadtxt(os.path.join(cv_dir, "y.txt"))

    i = -1
    while os.path.exists(os.path.join(cv_dir, "fold_%02d_train.txt" % (i + 1))):
        i += 1

        train = [int(x) for x in np.loadtxt(os.path.join(cv_dir, "fold_%02d_train.txt" % i))]

        trainX = X[train, :]
        trainy = y[train]

        fname = ".".join(["fold_%02d" % i, model_type])
        fullpath = os.path.join(cv_dir, fname)
        if os.path.exists(fullpath):
            print "model %s already exists, skipping..." % fullpath
            continue

        logfile_name = os.path.join(cv_dir, "fold_%02d_%s_train.log" % (i, model_type))
        logfile = open(logfile_name, 'w')
        print "training model", i
        with RedirectStdStreams(stdout=logfile, stderr=logfile):
            try:
                print "learning model"
                model = learn_model(X=trainX, y=trainy, **kwargs)
                print "learned"
            except KeyboardInterrupt:
                logfile.close()
                raise
            except Exception as e:
                print "Error training model:", str(e)
                continue
            model.save_trained_model(fullpath)
        logfile.close()


def cv_eval_models(cv_dir, model_type, load_model):
    X = np.loadtxt(os.path.join(cv_dir, "X.txt"))
    y = np.loadtxt(os.path.join(cv_dir, "y.txt"))

    residuals = []

    ii = 0
    while os.path.exists(os.path.join(cv_dir, "fold_%02d_train.txt" % ii)):
        train = [int(x) for x in np.loadtxt(os.path.join(cv_dir, "fold_%02d_train.txt" % ii))]
        test = [int(x) for x in np.loadtxt(os.path.join(cv_dir, "fold_%02d_test.txt" % ii))]

        testX = X[test, :]
        testy = y[test]

        fname = ".".join(["fold_%02d" % ii, model_type])
        model = load_model(os.path.join(cv_dir, fname))
        residuals += [model.predict(testX[i:i+1]) - testy[i] for i in range(len(testy))]
        ii += 1

    mean_abs_error = np.mean(np.abs(residuals))
    median_abs_error = np.median(np.abs(residuals))

    f_results = open(os.path.join(cv_dir, "%s_results.txt" % model_type), 'w')
    f_results.write('mean_abs_error %f\n' % mean_abs_error)
    f_results.write('median_abs_error %f\n' % median_abs_error)
    f_results.close()

    return mean_abs_error, median_abs_error
