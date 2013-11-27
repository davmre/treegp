import numpy as np
import unittest

from sparsegp.gp import GP, GPCov, optimize_gp_hyperparams
from sparsegp.features import featurizer_from_string

from sparsegp.cover_tree import VectorTree
import pyublas

class TestCTree(unittest.TestCase):

    def test_cover_tree_multiplication(self):

        # generate random data points
        np.random.seed(6)
        cluster = np.random.normal(size=(10, 2)) * 10
        cluster_locations = np.random.normal(size=(100, 2)) * 1000
        X = np.zeros((1000, 2))
        for (i,cl) in enumerate(cluster_locations):
            X[10*i:10*(i+1),:] = cluster + cl

        # build a cover tree
        dfn_param = np.array((1.0, 1.0), dtype=float)
        weight_param = np.array((1.0,), dtype=float)
        tree = VectorTree(X, 1, "euclidean", dfn_param, 'se', weight_param)

        # assign an arbitrary value to each data point
        v = np.array([1,2,3,4,5,6,7,8,9,10] * 100, dtype=float)
        tree.set_v(0, v)

        query_pt = np.matrix(cluster_locations[29,:], dtype=float, copy=True)

        #w = lambda x1, x2 : np.exp(-1 * np.linalg.norm(x1-x2, 2)**2 )
        #k = [w(query_pt, x) for x in X]
        #kv = np.dot(k, v)

        kv_tree = tree.weighted_sum(0, query_pt, 1e-4)
        self.assertAlmostEqual(0.893282181527, kv_tree, places=4)
        self.assertEqual(tree.fcalls, 54)


class TestGP(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [120, 30, 0, 1000, 32],
            [118, 31, 0, 1050, 32],
            [120, 29, 40, 1000, 34],
            [110, 30, 20, 3000, 34],
        ], dtype=float)
        self.y = np.array([
            -0.02,
            -0.01,
            -0.015,
            -0.005,
        ])
        self.evids = np.array([
            1,
            2,
            3,
            4,
        ])
        self.testX1 = np.array([[120, 30, 0, 1025, 32], ], dtype=float)
        self.testX2 = np.array([[119, 31, 0, 1000, 33], ], dtype=float)

        self.cov = GPCov(wfn_params=[.0187,], dfn_params=[ 9.00, 1.0], wfn_str="se", dfn_str="lld")
        self.noise_var = .022


        self.gp = GP(X=self.X, y=self.y, noise_var=self.noise_var, cov_main=self.cov, compute_ll=True, compute_grad=True)

    def test_sparse_gradient(self):
        g_sparse = self.gp._log_likelihood_gradient(None, None, None, self.gp.Kinv)
        g_dense = self.gp._log_likelihood_gradient(None, None, None, self.gp.Kinv.todense())
        self.assertTrue( (np.abs(g_sparse - g_dense) < 0.0001 ).all() )

    def test_SE_gradient(self):
        grad = self.gp.ll_grad

        nllgrad, x0, build_gp, _ = optimize_gp_hyperparams(X=self.X, y=self.y, noise_var=self.noise_var, cov_main=self.cov)

        n = len(x0)
        kp  = x0
        eps = 1e-4
        empirical_grad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            gp1 = build_gp(kp, compute_ll=True)
            l1 = gp1.log_likelihood()
            kp[i] += 2*eps
            gp2 = build_gp(kp, compute_ll=True)
            l2 = gp2.log_likelihood()
            kp[i] -= eps
            empirical_grad[i] = (l2 - l1)/ (2*eps)

        self.assertTrue( (np.abs(grad - empirical_grad) < 0.01 ).all() )


    """
    def test_GPsave(self):
        model = learn_model(self.X, self.y, model_type="gp_lld", target="coda_decay", sta='AAK', optim_params=construct_optim_params("'method': 'none', 'normalize': False"))
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)

        fname = "test_gp_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "gp_lld")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

        ll = nmodel.log_likelihood()
        ll1 = nmodel.log_p(cond=self.X, x=self.y)
        # unlike with the other models, these likelihoods are not
        # necessarily equal in the GP case

        s = nmodel.sample(self.X)
    """

class TestSemiParametric(unittest.TestCase):


    def setUp(self):
        N = 10
        x = np.linspace(-5,5,N)
        self.X = np.reshape(x, (-1, 1))

        self.basis = 'poly3'
        H, self.featurizer, self.featurizer_recovery = featurizer_from_string(self.X, self.basis, extract_dim=0)

        self.beta = [9.91898792e-01,  -1.62113090e+00,   3.15437605e+00,   1.25732838e+00]
        self.B = np.eye(4) * 9
        self.b = np.zeros((4,))

        p = np.dot(H, self.beta)

        def covar_matrix(x, k):
            n = len(x)
            K = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    K[i,j] = k(x[i], x[j])
            return K
        k1 = lambda x1, x2 : .001*np.exp( - ((x1-x2)/1.5)**2 ) + (.001 if x1==x2 else 0)
        K1 = covar_matrix(x, k1)

        np.random.seed(0)
        f1 = np.random.multivariate_normal(mean=np.zeros((len(x),)), cov=K1)
        self.y1 = f1 + p


        self.cov = GPCov(wfn_params=[.001,], dfn_params=[ 1.5,], wfn_str="se", dfn_str="euclidean")
        self.noise_var = .001

        self.gp = GP(X=self.X,
                     y=self.y1,
                     noise_var = self.noise_var,
                     cov_main = self.cov,
                     basis = self.basis,
                     featurizer_recovery = self.featurizer_recovery,
                     param_mean=self.b,
                     param_cov=self.B,
                     compute_ll=True,
                     compute_grad=True,
                     sparse_threshold=0,
                     build_tree=False)


    def test_param_recovery(self):
        gp = self.gp
        inferred_beta = gp.param_predict()
        self.assertTrue( ( np.abs(inferred_beta - self.beta) < .1 ).all() )

        # make sure the posterior covariance matrix is reasonable
        posterior_covar = gp.param_covariance()
        self.assertTrue( np.max(posterior_covar.flatten()) < 1e-2 )



    def test_likelihood(self):

        # in the limit of a prior forcing the parameters to be zero,
        # the semiparametric likelihood should match that of a
        # standard GP.
        gp_smallparam = GP(X=self.X, y=self.y1, noise_var=self.noise_var, cov_main=self.cov, dfn_str="euclidean", basis=self.basis, featurizer_recovery=self.featurizer_recovery, param_mean=self.b, param_cov=np.eye(len(self.b)) * 0.000000000000001, compute_ll=True, sparse_threshold=0)
        gp_noparam = GP(X=self.X, y=self.y1, noise_var=self.noise_var, cov_main=self.cov, dfn_str="euclidean", basis=None, compute_ll=True, sparse_threshold=0)

        self.assertGreater(self.gp.ll, gp_smallparam.ll)
        self.assertAlmostEqual(gp_smallparam.ll, gp_noparam.ll, places=-1)


    def test_gradient(self):
        grad = self.gp.ll_grad

        nllgrad, x0, build_gp, _ = optimize_gp_hyperparams(X=self.X, y=self.y1, basis=self.basis, featurizer_recovery=self.featurizer_recovery, param_mean=self.b, param_cov=self.B, noise_var=self.noise_var, cov_main=self.cov)

        n = len(x0)
        kp  = x0
        eps = 1e-6
        empirical_grad = np.zeros(n)
        for i in range(n):
            kp[i] -= eps
            gp1 = build_gp(kp, compute_ll=True)
            l1 = gp1.log_likelihood()
            kp[i] += 2*eps
            gp2 = build_gp(kp, compute_ll=True)
            l2 = gp2.log_likelihood()
            kp[i] -= eps
            empirical_grad[i] = (l2 - l1)/ (2*eps)

        self.assertTrue( (np.abs(grad - empirical_grad) < 0.01 ).all() )


    def test_load_save(self):
        gp1 = self.gp
        gp1.save_trained_model("test_semi.npz")
        gp2 = GP(fname="test_semi.npz", build_tree=False)

        pts = np.reshape(np.linspace(-5, 5, 20), (-1, 1))
        p1 = gp1.predict(pts)
        v1 = gp1.variance(pts)
        p2 = gp2.predict(pts)
        v2 = gp2.variance(pts)
        print p1, p2
        self.assertTrue((p1 == p2).all())
        self.assertTrue((v1 == v2).all())


if __name__ == '__main__':
    unittest.main()
