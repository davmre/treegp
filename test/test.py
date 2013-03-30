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
        self.gp = GaussianProcess(self.X, self.y, kernel = "se", kernel_params = self.start_params)

    def test_pred(self):
        #quantiles = munge.distance_quantiles(self.nX)

        x = self.gp.predict(self.X)
        v = self.gp.variance(self.X)
        ll = self.gp.posterior_log_likelihood(self.X[0, :], self.y[0])

    def test_cv_eval(self):
        loss = evaluate.test_kfold(self.X, self.y, folds=5, kernel="se_iso", kernel_params=(0.1, 1, 22.0))

    def test_learn_hyperparams(self):
        priors = [InvGamma(1.0, 1.0), InvGamma(1.0, 1.0), LogNormal(3.0, 2.0)]
        best_params, v = learn.learn_hyperparams(self.X, self.y, "se", start_kernel_params = self.start_params, kernel_priors=priors)

        print best_params

    def test_SE_gradient(self):

        kernel = "se"
        grad = learn.gp_grad(X=self.X, y=self.y, kernel=kernel, kernel_params=self.start_params, kernel_extra=None)

        n = len(self.start_params)
        kp  = self.start_params
        eps = 1e-4
        empirical_grad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            gp = GaussianProcess(X=self.X, y=self.y, kernel=kernel, kernel_params=kp)
            l1 = gp.log_likelihood()
            kp[i] += 2*eps
            gp = GaussianProcess(X=self.X, y=self.y, kernel=kernel, kernel_params=kp)
            l2 = gp.log_likelihood()
            kp[i] -= eps
            empirical_grad[i] = (l2 - l1)/ (2*eps)

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

if __name__ == '__main__':
    unittest.main()
