import numpy as np

import kernels
from gp import GaussianProcess

def sq_loss(y1, y2):
    return np.sum(np.abs(y1-y2)**2)

def rms_loss(y1, y2):
    return np.sqrt(np.mean((y1-y2)**2))

def abs_loss(y1, y2):
    return np.sum(np.abs(y1-y2))


def test_kfold(X, y, folds, kernel, K=None, loss_fn=sq_loss, train_loss=False):
    """
    Run kfold cross-validation to evaluate the GP mean predictions with regard to a specified loss function.

    K is an optional pre-computed kernel matrix.
    """

    n = X.shape[0]
    foldsize = np.floor(n/folds)

    loss = 0
    validate_n = 0
    baseline_loss = 0

    # compute the loss obtained by both training and testing on the entire dataset. this is obviously illegitimate.
    if train_loss or (K is None):
        K = kernel(X,X)
        gp = GaussianProcess(X=X, y=y, kernel=kernel, K=K)
        predictions = gp.predict(X)
        tl = loss_fn(predictions, y)
        print "train pred mean", np.mean(predictions), "true mean", np.mean(y)

    kfold_predictions = np.zeros((n,))

    for i in range(folds):
        foldstart = i*foldsize
        foldend = (i+1)*foldsize if i < folds-1 else n
        print "fold %f from %d to %d, size (%d)" % (i, foldstart, foldend, foldend-foldstart)

        # use all but one fold for training, and the remaining fold as
        # the test set
        train = np.concatenate([np.arange(0, foldstart, dtype=np.uint), np.arange(foldend, n, dtype=np.uint)])
        test = np.arange(foldstart, foldend, dtype=np.uint)

        # train the GP and evaluate its predictions
        gp = GaussianProcess(X=X[train, :], y=y[train,:], kernel=kernel, K=K[train, :][:, train])
        predictions = gp.predict(X[test,:])
        kfold_predictions[foldstart:foldend] = predictions
        loss += loss_fn(predictions, y[test])
        baseline_loss += loss_fn(np.mean(y[train,:]), y[test,:])
        print "pred mean", np.mean(predictions), "train mean", np.mean(y[train,:]), "test mean", np.mean(y[test,:])
        print "pred loss", loss_fn(predictions, y[test]), "baseline_loss", loss_fn(np.mean(y[train,:]), y[test,:])

    if train_loss:
        return loss/float(foldsize*folds), tl/float(n), baseline_loss/float(foldsize*folds), kfold_predictions
    else:
        return loss/float(n), kfold_predictions
