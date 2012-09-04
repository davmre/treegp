from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sys
import itertools, pickle, traceback

import numpy as np
import scipy.linalg, scipy.optimize
import kernels

def sq_loss(y1, y2):
    return np.sum(np.abs(y1-y2)**2)

def rms_loss(y1, y2):
    return np.sqrt(np.mean((y1-y2)**2))

def abs_loss(y1, y2):
    return np.sum(np.abs(y1-y2))


def log_det(X):
    w = np.linalg.eigvalsh(X)
    return np.sum(np.log(w))

    # do cross-validation to find the best hyperparameters
    # folds: do k-fold cross-validation. if folds=-1, do leave-one-out CV
    # kernel_params_values: a list of lists, with with ith list giving values to try for the ith kernel param
    # sigma_values: list of values to try for sigma
    # if either "values" param is None, don't optimize over that param
def cross_validate(X, y, kernel="se", folds=-1, kernel_params_values=None, kernel_extra=None, sigma_values=None):

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

def test_kfold(X, y, folds, kernel, kernel_params, kernel_extra, K, loss_fn, train_loss=False):
    n = X.shape[0]
    foldsize = np.floor(n/folds)

    loss = 0
    validate_n = 0
    baseline_loss = 0

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
        train = np.concatenate([np.arange(0, foldstart, dtype=np.uint), np.arange(foldend, n, dtype=np.uint)])
        validate = np.arange(foldstart, foldend, dtype=np.uint)
        gp = GaussianProcess(X=X[train, :], y=y[train,:], kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra, K=K[train, :][:, train], inv=False)
        predictions = gp.predict(X[validate,:])
        kfold_predictions[foldstart:foldend] = predictions
        loss += loss_fn(predictions, y[validate])
        baseline_loss += loss_fn(np.mean(y[train,:]), y[validate,:])
        print "pred mean", np.mean(predictions), "train mean", np.mean(y[train,:]), "validate mean", np.mean(y[validate,:])
        print "pred loss", loss_fn(predictions, y[validate]), "baseline_loss", loss_fn(np.mean(y[train,:]), y[validate,:])

    if train_loss:
        return loss/float(foldsize*folds), tl/float(n), baseline_loss/float(foldsize*folds), kfold_predictions
    else:
        return loss/float(n), kfold_predictions

class GaussianProcess:

    def __init__(self, X=None, y=None, kernel="se", kernel_params=(1, 1,), kernel_priors = None, kernel_extra = None, K=None, inv=True, mean="constant", ignore_pos_def_errors=False, fname=None):
        if fname is not None:
            self.load_trained_model(fname)
        else:
            self.kernel_name = kernel
            self.kernel_params = np.asarray(kernel_params)
            self.kernel_priors = kernel_priors
            self.kernel = kernels.setup_kernel(kernel, kernel_params, kernel_extra, kernel_priors)
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
            self.mean = mean

            # train model
            if K is None:
                self.K = self.kernel(X, X)
            else:
                self.K = K

            self.L = None
            self.alpha = None
            self.Kinv = None
            self.posdef = True
            try:
                self.L = scipy.linalg.cholesky(self.K, lower=True)
                self.alpha = scipy.linalg.cho_solve((self.L, True), self.y)
            except np.linalg.linalg.LinAlgError:

                if ignore_pos_def_errors:
                    self.posdef=False
                    return
                else:
                     raise

#                     u,v = np.linalg.eig(self.K)
#                     print self.K, u

#                     import pdb
#                     pdb.set_trace()
#                     sys.exit(1)
            except ValueError:
                raise

#                import pdb
#                pdb.set_trace()
#                self.K = self.kernel(X, X)

            if inv:
                self.__invert_kernel_matrix()

    def __invert_kernel_matrix(self):
        if self.Kinv is None and self.L is not None:
            invL = scipy.linalg.inv(self.L)
            self.Kinv = np.dot(invL.T, invL)

    def sample(self, X1):
        X1 = np.array(X1)
        if len(X1.shape) == 1:
            X1 = np.reshape(X1, (1, -1))

        (n,d) = X1.shape
        means = self.predict(X1)
        K = self.variance(X1)

        L = scipy.linalg.cholesky(K, lower=True)
        samples = np.random.randn(n, 1)
        samples = means + np.dot(L, samples)
        return samples

    def predict(self, X1):
        if not self.posdef:
            return self.mu

        K = self.kernel(self.X, X1)
        return self.mu + np.dot(K.T, self.alpha)

    def variance(self, X1):
        if not self.posdef:
            return 0

        self.__invert_kernel_matrix()
        K = self.kernel(self.X, X1)
        return self.kernel(X1,X1) - np.dot(K.T, np.dot(self.Kinv, K))

    def posterior_log_likelihood(self, X1, y):
        y = np.array(y)
        if len(y.shape) == 0:
            n = 1
        else:
            n = len(y)

        if not self.posdef:
            return np.float('-inf')

        self.__invert_kernel_matrix()

        K = self.variance(X1)
        y = y-self.predict(X1)

        if n==1:
            var = K[0,0]
            ll1 = - .5 * ((y)**2 / var + np.log(2*np.pi*var) )

        if K[0,0] < 0:
            import pdb
            pdb.set_trace()


        L = scipy.linalg.cholesky(K, lower=True)
        ld2 = np.log(np.diag(L)).sum()
        alpha = scipy.linalg.cho_solve((L, True), y)
        ll =  -.5 * (np.dot(y.T, alpha) + n * np.log(2*np.pi)) - ld2
        return ll


    def log_likelihood(self):
        if not self.posdef:
            return np.float('-inf')

        self.__invert_kernel_matrix()

        # the determinant of a symmetric pos. def. matrix is the product of squares of the diagonal elements of the Cholesky factor
        ld2 = np.log(np.diag(self.L)).sum()
        ll =  -.5 * (np.dot(self.y.T, self.alpha) + self.n * np.log(2*np.pi)) - ld2
#        print "params", self.kernel_params

#        print "returning ll %f = -.5 * (%f + %f + %f)" % (ll, np.dot(self.y.T, self.alpha), ld, self.n * np.log(2*np.pi))

        return ll

    def log_likelihood_gradient(self):
        grad = np.zeros(self.kernel_params.shape)

        if not self.posdef:
            return grad

        self.__invert_kernel_matrix()

        for i,p in enumerate(self.kernel_params):

            dKdi = self.kernel.derivative_wrt_i(i, self.X, self.X)
#            print "deriv wrt %d is" % (i), dKdi
            dlldi = .5 * np.dot(self.alpha.T, np.dot(dKdi, self.alpha))
#            dlldi1 = dlldi
            dlldi -= .5 * np.sum(np.sum(self.Kinv.T * dKdi))
#            print "dlldi is %f = .5 * %f - .5 * %f" % (dlldi, dlldi1*2, (dlldi1-dlldi)*2)
#            print "norm2 alpha is %f" % (np.dot(self.alpha,  self.alpha))
#            alt = .5 * np.dot(self.y.T, np.dot(self.Kinv, np.dot(dKdi, np.dot(self.Kinv, self.y))))
#            print "alt is %f" % (alt)

            grad[i] = dlldi
#        print self.kernel_params, "returning grad", grad
        return grad

    def save_trained_model(self, filename):
        kname = np.array((self.kernel_name,))
        mname = np.array((self.mean,))
        np.savez(filename, X = self.X, y=self.y, mu = np.array((self.mu,)), kernel_name=kname, kernel_params=self.kernel_params, mname = mname, alpha=self.alpha, Kinv=self.Kinv, K=self.K, L=self.L)

    def load_trained_model(self, filename):
        npzfile = np.load(filename)
        self.X = npzfile['X']
        self.y = npzfile['y']
        self.mu = npzfile['mu'][0]
        self.mean = npzfile['mname'][0]
        self.kernel_name = npzfile['kernel_name'][0]
        self.kernel_params = npzfile['kernel_params']
        self.alpha = npzfile['alpha']
        self.Kinv = npzfile['Kinv']
        self.L = npzfile['L']
        self.K = npzfile['K']

        self.n = self.X.shape[0]
        self.kernel = kernels.setup_kernel(kernel_name, kernel_params)
#    def validation_loss(self, trainIdx, valIdx, kernel_params, loss_fn):

def gp_ll(X, y, kernel, kernel_params, kernel_extra):
    try:
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra)
        ll = gp.log_likelihood()
    except:
        ll = np.float("-inf")
    return ll

def gp_grad(X, y, kernel, kernel_params, kernel_extra):
    try:
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra)
        grad = gp.log_likelihood_gradient()
    except:
        grad = np.zeros(kernel_params.shape)
    return grad

def gp_nll_ngrad(X, y, kernel, kernel_params, kernel_extra, kernel_priors):
    try:
        gp = GaussianProcess(X=X, y=y, kernel=kernel, kernel_params=kernel_params, kernel_extra=kernel_extra, kernel_priors=kernel_priors, ignore_pos_def_errors=True)
        nll = -1 * gp.log_likelihood()
        ngrad = -1 * gp.log_likelihood_gradient()

        npll = -1 * gp.kernel.param_prior_ll()
        npgrad = -1 * gp.kernel.param_prior_grad()

#        print "nll %f + %f = %f" % (nll, npll, nll+npll)

        nll += npll
        ngrad += npgrad

        print "optimizing params", kernel_params, "ll", -1 *nll, "grad", ngrad

    except np.linalg.linalg.LinAlgError:
        print "warning: lin alg error in likelihood computation (%f, %f), returning inf" % (nll, npll)
        nll = np.float("inf")
        ngrad = np.zeros(kernel_params.shape)

    return nll, ngrad

def optimize_hyperparams(X, y, kernel, start_kernel_params, kernel_extra=None, kernel_priors=None):
    ll = lambda params: -1 * gp_ll(X, y, kernel, params, kernel_extra)
    grad = lambda params: -1 * gp_grad(X, y, kernel, params, kernel_extra)
#    best_params = scipy.optimize.fmin_bfgs(f=ll, x0=start_kernel_params, fprime=grad)

    llgrad = lambda params: gp_nll_ngrad(X, y, kernel, params, kernel_extra, kernel_priors)

    skp = np.asarray(start_kernel_params)
    new_params = lambda  :  np.exp(np.log(skp) + np.random.randn(len(skp)) * 2)
    start_param_set = [skp,] + [new_params() for i in range(3)]

    print "start param set"
    print start_param_set

    best_params = skp
    nll, grad = llgrad(skp)
    best_cost = nll

    for params in start_param_set:

#    best_params, v, d = scipy.optimize.fmin_l_bfgs_b(func=llgrad, x0=start_kernel_params, bounds= [(1e-20, None),]*len(start_kernel_params))
        opt_params, v, d = scipy.optimize.fmin_l_bfgs_b(func=llgrad, x0=start_kernel_params, bounds= [(1e-20, None),]*len(start_kernel_params))

        if v < best_cost:
            best_cost = v
            best_params = opt_params

#    print "start ll", ll(start_kernel_params)
#    print "best ll", ll(best_params)

    print "OPTIMZIATION FINISHED: found best params", best_params
    print "ll", v
    return best_params, v

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


def gp_1D_predict(
    gp,
    num_steps = 100,
    x_min = None,
    x_max = None,
    ):
    """Predict and plot the GP's predictions over the range provided"""

    from pylab import figure, plot, show
    # print gp.X
    if None == x_min: x_min = min( x[0] for x in gp.X )
    if None == x_max: x_max = max( x[0] for x in gp.X )
    x_max = float( x_max )
    x_min = float( x_min )
    test_x = np.reshape(np.linspace( x_min, x_max, num_steps ), (-1, 1))
    print test_x.shape, gp.X.shape
    mean = gp.predict( test_x )
    print "mean", mean
    var = np.diag(gp.variance( test_x ))
    print "var", var
    figure()
    gp_plot_prediction( test_x, mean, var )
    show()

def gp_plot_prediction(predict_x, mean, variance = None):
    """
    Plot a gp's prediction using pylab including error bars if variance specified

    Error bars are 2 * standard_deviation as in GP for ML book
    """
    from pylab import plot, concatenate, fill
    if None != variance:
        # check variances are just about +ve - could signify a bug if not
        #assert variance.all() > -1e-10
        data = [
            (x,y,max(v,0.0))
            for x,y,v
            in zip( predict_x, mean.flat, variance  )
            ]
    else:
        data = [
            (x,y)
            for x,y
            in zip( predict_x, mean )
            ]
    data.sort( key = lambda d: d[0] ) # sort on X axis
    predict_x = [ d[0] for d in data ]
    predict_y = np.array( [ d[1] for d in data ] )
    plot( predict_x, predict_y, color='k', linestyle=':' )
    if None != variance:
        sd = np.sqrt( np.array( [ d[2] for d in data ] ) )
        var_x = concatenate((predict_x, predict_x[::-1]))
        var_y = concatenate((predict_y + 2.0 * sd, (predict_y - 2.0 * sd)[::-1]))
        p = fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3')

def main():
    X = np.array(((0,0), (0,1), (1,0), (1,1)))
    y = np.array((1, 50, 50, 15))


    gp = GaussianProcess(X=X, y=y, kernel="sqexp", kernel_params=(1,), sigma=0.01)
    print gp.predict((0,0))

#    print pickle.dumps(gp)

#    plot_interpolated_surface(gp, X, y)

if __name__ == "__main__":
    main()

