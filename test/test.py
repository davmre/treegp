import unittest
import os
import numpy as np

from gpr import munge, kernels, evaluate, learn, distributions, plot
from gpr.gp import GaussianProcess
from gpr.distributions import InvGamma, LogNormal

"""
class TestAbalone(unittest.TestCase):

    def setUp(self):
        pass

    def test_abalone_preprocess(self):

        # load abalone data, representing gender as integer values
        convertfunc = lambda s: float(0.0) if s=='M' else float(1.0) if s=='F' else float(2.0) if s=='I' else float(-1)
        data = np.genfromtxt("test/abalone.data", delimiter=',', converters={0:convertfunc})
        data = data.view(float).reshape((data.shape[0], -1))

        idata, d = munge.categorical_to_indicators(data, 0)

        X,y,_,_ = munge.preprocess(idata, target=(idata.shape[1]-1))

        self.X = X
        self.y = y
"""

class TestSimple(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
                [ -4.0 ],
                [ -3.0 ],
                [ -1.0 ],
                [  0.0 ],
                [  2.0 ],
                ])
        self.y = np.array( [
                -2.0,
                 0.0,
                 1.0,
                 2.0,
                 -1.0
                 ] )

        self.start_params = np.array((.2, 1, 1))

        priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3.0, 2.0)]
        self.k = kernels.setup_kernel(name="se", params = self.start_params, priors=priors)
        self.gp = GaussianProcess(self.X, self.y, kernel = self.k)

    def test_pred(self):
        #quantiles = munge.distance_quantiles(self.nX)

        x = self.gp.predict(self.X)
        v = self.gp.variance(self.X)
        ll = self.gp.posterior_log_likelihood(self.X[0, :], self.y[0])

    def test_cv_eval(self):
        k_iso = kernels.setup_kernel(name="se_iso", params = (0.1, 1, 22.0))
        loss = evaluate.test_kfold(self.X, self.y, folds=5, kernel=k_iso)

    def test_learn_hyperparams(self):
        best_params, v = learn.learn_hyperparams(self.X, self.y, kernel=self.k, start_kernel_params = self.start_params)
        print best_params

    def test_SE_gradient(self):
        grad = learn.gp_grad(X=self.X, y=self.y, kernel=self.k, kernel_params=self.start_params)

        n = len(self.start_params)
        kp  = self.start_params
        eps = 1e-4
        empirical_grad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            self.k.set_params(kp)
            gp = GaussianProcess(X=self.X, y=self.y, kernel=self.k, compute_ll=True)
            l1 = gp.log_likelihood()
            kp[i] += 2*eps
            self.k.set_params(kp)
            gp = GaussianProcess(X=self.X, y=self.y, kernel=self.k, compute_ll=True)
            l2 = gp.log_likelihood()
            kp[i] -= eps
            self.k.set_params(kp)
            empirical_grad[i] = (l2 - l1)/ (2*eps)

        print grad
        print empirical_grad
        self.assertTrue( (np.abs(grad - empirical_grad) < 0.01 ).all() )

    def test_prior_gradient(self):
        kernel = "se"
        k = kernels.setup_kernel(name=kernel, params=self.start_params, extra=None, priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3.0, 2.0)])
        pgrad = k.param_prior_grad()

        n = len(self.start_params)
        kp  = self.start_params
        eps = 1e-4
        empirical_pgrad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            k = kernels.setup_kernel(name=kernel, params=kp, extra=None, priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3.0, 2.0)])
            l1 = k.param_prior_ll()
            kp[i] += 2*eps
            k = kernels.setup_kernel(name=kernel, params=kp, extra=None, priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3.0, 2.0)])
            l2 = k.param_prior_ll()
            kp[i] -= eps
            empirical_pgrad[i] = (l2 - l1)/ (2*eps)

        self.assertTrue( (np.abs(pgrad - empirical_pgrad) < 0.01 ).all() )

    def test_get_params(self):
        p = np.array(self.k.get_params())
        self.assertTrue( (p == self.start_params).all() )

    def test_plot(self):
        plot.predict_1d(self.gp, x_min = -5.0, x_max = 5.0)

    def test_load_save(self):
        gp1 = self.gp
        gp1.save_trained_model("test_toy.npz")
        gp2 = GaussianProcess(fname="test_toy.npz")

        pts = np.linspace(-5, 5, 20)
        p1 = gp1.predict(pts)
        v1 = gp1.variance(pts)
        p2 = gp2.predict(pts)
        v2 = gp2.variance(pts)
        self.assertTrue((p1 == p2).all())
        self.assertTrue((v1 == v2).all())

    def tearDown(self):
        try:
            os.remove("test_toy.npz")
        except OSError:
            pass


class TestSemiParametric(unittest.TestCase):


    def setUp(self):

        N = 100
        x = np.linspace(-5,5,N)
        self.X = np.reshape(x, (-1, 1))
        H = np.vstack([x**3, x**2, x, np.ones(N)]) # basis functions for a cubic regression
        self.beta = [9.91898792e-01,  -1.62113090e+00,   3.15437605e+00,   1.25732838e+00]
        self.B = np.eye(4) * 9
        self.b = np.zeros((4,))

        p = np.dot(H.T, self.beta)

        def covar_matrix(x, k):
            n = len(x)
            K = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    K[i,j] = k(x[i], x[j])
            return K
        k1 = lambda x1, x2 : .001*np.exp( - ((x1-x2)/.01)**2 ) + (.001 if x1==x2 else 0)
        K1 = covar_matrix(x, k1)

        np.random.seed(0)
        f1 = np.random.multivariate_normal(mean=np.zeros((len(x),)), cov=K1)
        self.y1 = f1 + p

        self.basisfns = [lambda x : x**3, lambda x : x**2, lambda x : x, lambda x : 1]


        self.k = kernels.setup_kernel(name="se", params = np.array((.001, .01, .001)))
        self.gp = GaussianProcess(X=self.X[::10,:],
                                  y=self.y1[::10],
                                  kernel=self.k,
                                  mean="parametric",
                                  basisfns=self.basisfns,
                                  param_mean=self.b,
                                  param_cov=self.B,
                                  compute_ll=True,
                                  compute_grad=True)


    def test_param_recovery(self):


        gp = self.gp
        inferred_beta = gp.param_predict()
        self.assertTrue( ( np.abs(inferred_beta - self.beta) < .1 ).all() )

        # make sure the posterior covariance matrix is reasonable
        posterior_covar = gp.param_covariance()
        self.assertTrue( np.max(posterior_covar.flatten()) < 1e-2 )

        # we should have the most uncertainty about the low-order
        # params (e.g. the constant term), and the least uncertainty
        # about the high-order terms, since a small change in a high-
        # order term has a larger effect on the resulting function.
        posterior_var = np.diag(posterior_covar)
        self.assertTrue( posterior_var[3] > posterior_var[2] )
        self.assertTrue( posterior_var[2] > posterior_var[1] )
        self.assertTrue( posterior_var[1] > posterior_var[0] )


    def test_likelihood(self):

        # in the limit of a prior forcing the parameters to be zero,
        # the semiparametric likelihood should match that of a
        # standard GP.

        gp = self.gp

        gp_smallparam = GaussianProcess(X=self.X[::10,:], y=self.y1[::10], kernel=self.k, mean="parametric", basisfns=self.basisfns, param_mean=self.b, param_cov=np.eye(4) * 0.0000000000001, compute_ll=True)
        gp_noparam = GaussianProcess(X=self.X[::10,:], y=self.y1[::10], kernel=self.k, mean="zero", compute_ll=True)

        self.assertGreater(self.gp.ll, gp_smallparam.ll)
        self.assertAlmostEqual(gp_smallparam.ll, gp_noparam.ll, places=-1)

    def test_gradient(self):
        gp = self.gp
        k0 = kernels.setup_kernel("se", params=np.array((.0011, .01, .001)))
        k1 = kernels.setup_kernel("se", params=np.array((.001, .0101, .001)))
        k2 = kernels.setup_kernel("se", params=np.array((.001, .01, .0011)))
        gp_grad0 = GaussianProcess(X=self.X[::10,:], y=self.y1[::10], kernel=k0, mean="parametric", basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, compute_ll=True)
        gp_grad1 = GaussianProcess(X=self.X[::10,:], y=self.y1[::10], kernel=k1, mean="parametric", basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, compute_ll=True)
        gp_grad2 = GaussianProcess(X=self.X[::10,:], y=self.y1[::10], kernel=k2, mean="parametric", basisfns=self.basisfns, param_mean=self.b, param_cov=self.B, compute_ll=True)

        empirical_llgrad = [(gp_grad0.ll - gp.ll) / .0001,
                            (gp_grad1.ll - gp.ll) / .0001,
                            (gp_grad2.ll - gp.ll) / .0001]

        self.assertTrue( ( np.abs(empirical_llgrad - gp.ll_grad) < 1 ).all() )


    def test_load_save(self):
        gp1 = self.gp
        gp1.save_trained_model("test_semi.npz")
        gp2 = GaussianProcess(fname="test_semi.npz")

        pts = np.linspace(-5, 5, 20)
        p1 = gp1.predict(pts)
        v1 = gp1.variance(pts)
        p2 = gp2.predict(pts)
        v2 = gp2.variance(pts)
        self.assertTrue((p1 == p2).all())
        self.assertTrue((v1 == v2).all())


if __name__ == '__main__':
    unittest.main()
