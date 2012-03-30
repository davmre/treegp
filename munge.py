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

def detect_indicator_cols(X):
    icols = []
    for col in range(X.shape[1]):
        indicator = True
        for val in X[:, col]:
            if val != 0 and val != 1 and val != -1:
                indicator = False
                break
        if indicator:
            icols.append(col)
    return icols

def center_mean(X, icols = None):

    if icols is None:
        icols = detect_indicator_cols(X)
    means = np.mean(X, axis=0)

    if len(icols) > 0:
        means[icols] = 0
    
    return X-means, means, icols

def normalize_variance(X, icols=None):
    
    if icols is None:
        icols = detect_indicator_cols(X)
    std = np.std(X, axis=0)

    if len(icols) > 0:
        std[icols] = 1
    
    return X/std, std, icols

def preprocess(data, target):

    p = np.random.permutation(data.shape[0])
    pdata = data[p , :]

    try:
        for t in target:
            pass
    except TypeError:
        target = [target,]
    not_target = sorted(list(set(range(data.shape[1])) - set(target)))

    X = pdata[:, not_target]
    y = pdata[:, target]
    
    cX, Xmeans, icols = center_mean(X)
    nX, Xstd, icols = normalize_variance(X, icols)
    
    cy, ymean, icols = center_mean(y, [])
    ny, ystd, icols = normalize_variance(y, [])

    return X, y, nX, ny

def distance_quantiles(X, quantiles, samples=1000):
    n = X.shape[0]
    distances = np.sort([np.linalg.norm(X[np.random.randint(0, n):,] - X[np.random.randint(0, n), :]) for i in range(samples)])
    indices = np.array(np.array(quantiles)*samples - 1, dtype=np.uint32)
    return distances[indices]

