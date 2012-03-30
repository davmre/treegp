from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import itertools
import pickle

import numpy as np
import scipy.linalg
import kernels

def sq_loss(y1, y2):
    return np.sum(.5 * np.abs(y1-y2)**2)

    # do cross-validation to find the best hyperparameters
    # folds: do k-fold cross-validation. if folds=-1, do leave-one-out CV
    # kernel_params_values: a list of lists, with with ith list giving values to try for the ith kernel param
    # sigma_values: list of values to try for sigma
    # if either "values" param is None, don't optimize over that param
def cross_validate(X, y, kernel="se", folds=-1, kernel_params_values=None, sigma_values=None):

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
            loss, train_loss, baseline_loss = test_kfold(X, y, folds, "se_noiseless", params, Ks, sq_loss, train_loss=True)
            print "give loss", loss, "training loss", train_loss, "baseline", baseline_loss
            f.write("%f %f %f %f\n" % (params[0], sigma, loss, train_loss))
            f.flush()
    f.close()

def test_kfold(X, y, folds, kernel, kernel_params, K, loss_fn, train_loss=False):
    n = X.shape[0]
    foldsize = np.floor(n/folds)

    loss = 0
    validate_n = 0
    baseline_loss = 0

    if train_loss:
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, K=K, inv=False)
        predictions = gp.predict(X)
        tl = loss_fn(predictions, y)

    for i in range(folds):
        train = np.concatenate([np.arange(0, i*foldsize, dtype=np.uint), np.arange((i+1)*foldsize, n, dtype=np.uint)])
        validate = np.arange(i*foldsize, (i+1)*foldsize, dtype=np.uint)
        gp = GaussianProcess(X=X[train, :], y=y[train,:], kernel=kernel, kernel_params=kernel_params, K=K[train, :][:, train], inv=False)
        predictions = gp.predict(X[validate,:])
        loss += loss_fn(predictions, y[validate])
        baseline_loss += loss_fn(np.mean(y[train,:]), y[validate,:])

    if train_loss:
        return loss/float(foldsize*folds), tl/float(n), baseline_loss/float(n)
    else:
        return loss/float(n)

class GaussianProcess:

    def __init__(self, fname=None, X=None, y=None, kernel="se", kernel_params=(1, 1,), K=None, inv=True, mean="constant"):
        if fname is not None:
            self.load_trained_model(fname)
        else:
            self.kernel_name = kernel
            self.kernel_params = kernel_params
            self.kernel = kernels.setup_kernel(kernel, kernel_params)
            self.X = X
            self.n = X.shape[0]

            if mean == "zero":
                self.mu = 0
                self.y = y 
            if mean == "constant":
                self.mu = np.mean(y)
                self.y = y - self.mu
            if mean == "linear":
                raise RuntimeError("linear mean not yet implemented...")

            # train model
            if K is None:
                self.K = self.kernel(X, X)
            else:
                self.K = K

#            print "got kernel matrix, decomposing..."
            self.L = scipy.linalg.cholesky(K, lower=True)
#            print "solving for alpha..."
            self.alpha = scipy.linalg.cho_solve((self.L, True), self.y)
            alpha2 = np.linalg.solve(self.L.T, np.linalg.solve(self.L,self.y))
            self.invKs = None
            if inv:
#                print "computing and storing inverse matrix..."
                self.__invert_kernel_matrix()
#            print "done."

    def __invert_kernel_matrix(self):
        if self.invKs is None:
            invL = scipy.linalg.inv(self.L)
            self.invKs = np.dot(invL.T, invL)
        
    def predict(self, X1):
        K = self.kernel(self.X, X1)
        return self.mu + np.dot(K.T, self.alpha)

    def variance(self, x):
        self.__invert_kernel_matrix()
        return self.kernel(x,x) - np.dot(k.T, np.dot(self.invKs, k))

    def training_set_loss(self, loss_fn=sq_loss, trials=None):
        if trials is None:
            trials = self.n

        total_loss=0
        for i in range(trials):
            p = self.predict(self.X[i,:])
            total_loss += loss_fn(self.y[i]+self.mu, p)
        return total_loss/self.n

    def save_trained_model(self, filename):
        kname = np.array((self.kernel_name,))
        np.savez(filename, X = self.X, y=self.y, mu = np.array((self.mu,)), sigma = np.array((self.sigma,)), kernel_name=kname, kernel_params=self.kernel_params, alpha=self.alpha, invKs=self.invKs, K=self.K, L=self.L)

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.X = npzfile['X']
        self.y = npzfile['y']
        self.mu = npzfile['mu'][0]
        self.sigma = npzfile['sigma'][0]
        self.kernel_name = npzfile['kernel_name'][0]
        self.kernel_params = npzfile['kernel_params']
        self.alpha = npzfile['alpha']
        self.invKs = npzfile['invKs']
        self.L = npzfile['L']
        self.K = npzfile['K']

        self.n = self.X.shape[0]
        self.kernel = kernels.setup_kernel(kernel_name, kernel_params)        
#    def validation_loss(self, trainIdx, valIdx, kernel_params, loss_fn):
        
                

def plot_interpolated_surface(gp, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xmin = np.min(X, 0)
    xmax = np.max(X, 0)

    u = np.linspace(xmin[0], xmax[0], 20)
    v = np.linspace(xmin[1], xmax[1], 20)

    xc = np.outer(np.ones((20,)), u)
    yc = np.outer(v, np.ones((20,)))

    k = np.zeros(xc.shape)
    for i in range(xc.shape[0]):
        for j in range(xc.shape[1]):
            k[i,j] = gp.predict((xc[i,j], yc[i,j]))

    #print xmin, xmax
    #print u, v
    #print x, y, k
    
    ax.plot_surface(xc, yc, k,  color='b')

    plt.show()

def main():
    X = np.array(((0,0), (0,1), (1,0), (1,1)))
    y = np.array((1, 50, 50, 15))


    gp = GaussianProcess(X=X, y=y, kernel="sqexp", kernel_params=(1,), sigma=0.01)
    print gp.predict((0,0))

#    print pickle.dumps(gp)

#    plot_interpolated_surface(gp, X, y)




if __name__ == "__main__":
    main()
