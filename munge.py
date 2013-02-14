import numpy as np

def categorical_to_indicators(X, col):

    vals = np.unique(X[:, col])
    d = dict()
    for i, val in enumerate(vals):
        d[val] = i

    Y = np.hstack([X[:, 0:col],  np.zeros((X.shape[0], len(vals))), X[:, col+1:]])
    for i, row in enumerate(Y):
        row[col + d[X[i, col]]] = 1

    return Y, d

def detect_indicator_cols(data):
    zeros = np.zeros(data.shape)
    ones = np.ones(data.shape)
    bools = ((data == zeros) + (data == ones)).all(axis=0)
    # TODO: look up the official way to do this
    return filter(lambda n: bools[n], range(len(bools)))

def center_mean(X, icols = []):

    means = np.mean(X, axis=0)

    if len(icols) > 0:
        means[icols] = 0
    
    return X-means, means

def normalize_variance(X, icols=[]):
    
    std = np.std(X, axis=0)

    if len(icols) > 0:
        std[icols] = 1
    
    return X/std, std

def preprocess(data, target):
    """
    Standard preprocessing for regression data: randomly permute the training points and rescale each feature to zero mean and unit variance. Splits the features of each training point into the inputs (X) and the targets (y).
    """

    p = np.random.permutation(data.shape[0])
    pdata = data[p , :]

    # detect boolean features and don't recenter/rescale them
    icols = detect_indicator_cols(data)

    # do the recentering/rescaling
    cdata, data_means = center_mean(pdata, icols)
    ndata, data_std = normalize_variance(pdata, icols)

    # pull out the X and y values
    try:
        for t in target:
            pass
    except TypeError:
        target = [target,]
    not_target = sorted(list(set(range(data.shape[1])) - set(target)))

    X = pdata[:, not_target]
    y = pdata[:, target]
    nX = ndata[:, not_target]
    ny = ndata[:, target]
    
    return nX, ny, X, y

def distance_quantiles(X, quantiles = (0.05, .1, 0.5, .9, 0.95), samples=1000):
    """
    Compute quantiles of the distribution of pairwise distances
    between data points. This often gives useful heuristic choices for
    the characteristic length scale of a regression kernel.
    """

    n = X.shape[0]
    distances = np.sort([np.linalg.norm(X[np.random.randint(0, n):,] - X[np.random.randint(0, n), :]) for i in range(samples)])
    indices = np.array(np.array(quantiles)*samples - 1, dtype=np.uint32)
    return distances[indices]

