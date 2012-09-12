import numpy as np

from gp import GaussianProcess

def sq_loss(y1, y2):
    return np.sum(np.abs(y1-y2)**2)

def rms_loss(y1, y2):
    return np.sqrt(np.mean((y1-y2)**2))

def abs_loss(y1, y2):
    return np.sum(np.abs(y1-y2))


def cross_validate(X, y, kernel="se", folds=-1, kernel_params_values=None, kernel_extra=None, sigma_values=None):
    """
     do cross-validation to find the best hyperparameters
     folds: do k-fold cross-validation. if folds=-1, do leave-one-out CV
     kernel_params_values: a list of lists, with with ith list giving values to try for the ith kernel param
     sigma_values: list of values to try for sigma
     if either "values" param is None, don't optimize over that param
     """

    f = open("cv_results.log", 'a')

    if folds==-1:
        folds = X.shape[0]

    best_params = np.ones((len(kernel_params_values),))
    best_sigma = 1
    best_param_loss = np.float("inf")

    for params in itertools.product(*kernel_params_values):
        se_kernel = kernels.SEKernel(params)
        K = se_kernel(X, X)
        for sigma in sigma_values:
            Ks = K + np.eye(K.shape[0])*(sigma**2)
            print params[0], sigma,
            loss, train_loss, baseline_loss = test_kfold(X, y, folds, "se_noiseless", params, kernel_extra, Ks, sq_loss, train_loss=True)
            print "give loss", loss, "training loss", train_loss, "baseline", baseline_loss
            f.write("%f %f %f %f\n" % (params[0], sigma, loss, train_loss))
            f.flush()
    f.close()

def test_kfold(X, y, folds, kernel, kernel_params, kernel_extra=None, K=None, loss_fn=sq_loss, train_loss=False):
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
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra, K=K, inv=False)
        predictions = gp.predict(X)
        tl = loss_fn(predictions, y)
        print "train pred mean", np.mean(predictions), "true mean", np.mean(y)
        K = gp.K

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
        gp = GaussianProcess(X=X[train, :], y=y[train,:], kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra, K=K[train, :][:, train], inv=False)
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
