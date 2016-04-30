import numpy as np


def ortho_poly_fit(x, degree = 1):
    n = degree + 1
    x = np.asarray(x).flatten()
    if(degree >= len(np.unique(x))):
            stop("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n)) #outer(x, seq_len(n) - 1, "^")
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)
    Z[:,0] = 1
    return Z, norm2, alpha

def ortho_poly_predict(x, alpha, norm2, degree=1):
    x = np.asarray(x).flatten()
    n = degree + 1
    Z = np.empty((len(x), n))
    Z[:,0] = 1
    if degree > 0:
        Z[:, 1] = x - alpha[0]
    if degree > 1:
      for i in np.arange(1,degree):
          Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
    Z /= np.sqrt(norm2)
    Z[:,0] = 1
    return Z

def sin_transform(X, n):
    X = np.asarray(X).flatten()
    features = np.empty((X.shape[0], 2*n+2))
    features[:, 0] = 1
    #features[:, 1] = np.log(X+100)
    features[:,1] = (X - 4000) / 1000.0
    for i in range(0,n):
        features[:, 2*i+2] = np.sin(X * (2*np.pi*(i+1))/15000.0)
        features[:, 2*i+3] = np.cos(X * (2*np.pi*(i+1))/15000.0)
    return features

def multilinear_featurizer(X, dims, means, scales):
    features = np.zeros((X.shape[0], len(dims)+1))
    features[:,0] = 1
    features[:, 1:] = (X[:, dims] - means)/scales
    return features

def build_multilinear(X, dims, transpose):
    Z = X[:,dims]
    means = np.mean(Z, axis=0)
    scales = np.std(Z, axis=0)
    if transpose:
        return lambda X: multilinear_featurizer(X, dims, means, scales).T, means, scales
    else:
        return lambda X: multilinear_featurizer(X, dims, means, scales), means, scales

def build_ortho_poly_featurizer(X, extract_dim, degree, transpose):
    Z, norm2, alpha = ortho_poly_fit(X[:,extract_dim], degree = degree)
    if transpose:
        return Z.T, lambda X : ortho_poly_predict(X[:,extract_dim], alpha, norm2, degree=degree).T, norm2, alpha
    else:
        return Z, lambda X : ortho_poly_predict(X[:,extract_dim], alpha, norm2, degree=degree), norm2, alpha

def build_sin_featurizer(extract_dim, degree, transpose):
    if transpose:
        return lambda X : sin_transform(X[:,extract_dim], degree).T
    else:
        return lambda X : sin_transform(X[:,extract_dim], degree)

def featurizer_from_string(X, desc, extract_dim=0, transpose=False):
    if desc.startswith("poly"):
        degree = int(desc[4:])
        Z, f, norm2, alpha = build_ortho_poly_featurizer(X, extract_dim, degree, transpose)
        return Z, f, {'norm2': norm2, 'alpha': alpha, 'extract_dim': [extract_dim,]}
    elif desc.startswith("sin"):
        degree = int(desc[3:])
        featurizer = build_sin_featurizer(extract_dim, degree, transpose)
        Z = featurizer(X)
        return Z, featurizer, {'extract_dim': [extract_dim,]}
    elif desc.startswith("mlinear"):
        featurizer, means,scales = build_multilinear(X, extract_dim, transpose)
        Z = featurizer(X)
        return Z, featurizer, {'means': means, 'scales': scales, 'extract_dim': extract_dim}
    else:
        raise ValueError("unrecognized feature type %s" % desc)

def recover_featurizer(desc, recovery_info, transpose=False):
    if desc.startswith("poly"):
        degree = int(desc[4:])
        norm2 = recovery_info['norm2']
        alpha = recovery_info['alpha']
        extract_dim = recovery_info['extract_dim'][0]
        if transpose:
            f = lambda X : ortho_poly_predict(X[:,extract_dim], alpha, norm2, degree=degree).T
        else:
            f = lambda X : ortho_poly_predict(X[:,extract_dim], alpha, norm2, degree=degree)
        return f, {'norm2': norm2, 'alpha': alpha, 'extract_dim': [extract_dim,]}
    elif desc.startswith("sin"):
        degree = int(desc[3:])
        extract_dim = recovery_info['extract_dim'][0]
        featurizer = build_sin_featurizer(extract_dim, degree, transpose)
        return featurizer, {'extract_dim': [extract_dim,]}
    elif desc.startswith("mlinear"):
        means = recovery_info['means']
        scales = recovery_info['scales']
        dims = recovery_info['extract_dim']
        if transpose:
            featurizer = lambda X: multilinear_featurizer(X, dims, means, scales).T
        else:
            featurizer = lambda X: multilinear_featurizer(X, dims, means, scales)
        return featurizer, {'means': means, 'scales': scales, 'extract_dim': dims}

    else:
        raise ValueError("unrecognized feature type %s" % desc)
