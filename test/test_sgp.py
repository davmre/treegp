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
            [120, 30, 0, ],
            [118, 31, 0, ],
            [120, 29, 40,],
            [110, 30, 20,],
        ], dtype=float)
        self.y = np.array([
            -2.0,
            -1.0,
            -1.5,
            -0.5,
        ])

        self.testX1 = np.array([[120, 30, 0,], ], dtype=float)
        self.testX2 = np.array([[119, 31, 0,], ], dtype=float)

        self.cov = GPCov(wfn_params=[2.0,], dfn_params=[ 900.00, 1000.0, ], wfn_str="compact2", dfn_str="lld")
        self.noise_var = .22


    def _check_gradient(self, cov, eps=1e-8):
        gp = GP(X=self.X, y=self.y, noise_var=self.noise_var, cov_main=cov, compute_ll=True, compute_grad=True)

        grad = gp.ll_grad
        nllgrad, x0, bounds, build_gp, _ = optimize_gp_hyperparams(X=self.X, y=self.y, noise_var=self.noise_var, cov_main=cov, sparse_invert=False)

        n = len(x0)
        kp  = x0
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

        self.assertTrue( (np.abs(grad - empirical_grad) < 0.00001 ).all() )

    def test_sparse_gradient(self):

        gp = GP(X=self.X, y=self.y, noise_var=self.noise_var, cov_main=self.cov, compute_ll=True, compute_grad=True)
        g_sparse = gp._log_likelihood_gradient(None, gp.Kinv)
        g_dense = gp._log_likelihood_gradient(None, gp.Kinv.todense())
        self.assertTrue( (np.abs(g_sparse - g_dense) < 0.0001 ).all() )

    def test_SE_gradient(self):

        cov1 = GPCov(wfn_params=[3.0,], dfn_params=[ 900.00, 1000.0, ], wfn_str="se", dfn_str="lld")
        self._check_gradient(cov1)

        cov2 = GPCov(wfn_params=[3.0,], dfn_params=[ 900.00, 1000.0, ], wfn_str="compact2", dfn_str="lld")
        self._check_gradient(cov2)

        cov3 = GPCov(wfn_params=[3.0,], dfn_params=[ 10.00, 10.0, 40.0], wfn_str="se", dfn_str="euclidean")
        self._check_gradient(cov3)



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
                     build_tree=False,
                     sparse_invert=True)


    def test_param_recovery(self):
        gp = self.gp
        inferred_beta = gp.param_mean()
        self.assertTrue( ( np.abs(inferred_beta - self.beta) < .1 ).all() )

        # make sure the posterior covariance matrix is reasonable
        posterior_covar = gp.param_covariance()
        self.assertTrue( np.max(posterior_covar.flatten()) < 1e-2 )



    def test_likelihood(self):

        # in the limit of a prior forcing the parameters to be zero,
        # the semiparametric likelihood should match that of a
        # standard GP.
        gp_smallparam = GP(X=self.X, y=self.y1, noise_var=self.noise_var, cov_main=self.cov, basis=self.basis, featurizer_recovery=self.featurizer_recovery, param_mean=self.b, param_cov=np.eye(len(self.b)) * 0.000000000000001, compute_ll=True, sparse_threshold=0)
        gp_noparam = GP(X=self.X, y=self.y1, noise_var=self.noise_var, cov_main=self.cov, basis=None, compute_ll=True, sparse_threshold=0)

        self.assertGreater(self.gp.ll, gp_smallparam.ll)
        self.assertAlmostEqual(gp_smallparam.ll, gp_noparam.ll, places=-1)


    def test_gradient(self):
        grad = self.gp.ll_grad

        nllgrad, x0, bounds, build_gp, _ = optimize_gp_hyperparams(X=self.X, y=self.y1, basis=self.basis, featurizer_recovery=self.featurizer_recovery, param_mean=self.b, param_cov=self.B, noise_var=self.noise_var, cov_main=self.cov)

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

        self.assertTrue( (np.abs(grad - empirical_grad) < 0.001 ).all() )

    def test_sparse_gradient(self):
        g_sparse = self.gp._log_likelihood_gradient(self.y1, self.gp.Kinv)
        g_dense = self.gp._log_likelihood_gradient(self.y1, self.gp.Kinv.todense())
        self.assertTrue( (np.abs(g_sparse - g_dense) < 0.00001 ).all() )


    def test_load_save(self):
        gp1 = self.gp
        gp1.save_trained_model("test_semi.npz")
        gp2 = GP(fname="test_semi.npz", build_tree=False)

        pts = np.reshape(np.linspace(-5, 5, 20), (-1, 1))
        p1 = gp1.predict(pts)
        v1 = gp1.variance(pts)
        p2 = gp2.predict(pts)
        v2 = gp2.variance(pts)

        self.assertTrue((p1 == p2).all())
        self.assertTrue((v1 == v2).all())


class TestCSFIC(unittest.TestCase):



    def setUp(self):
        N = 25
        x = np.linspace(-5,5,N)
        self.X = np.reshape(x, (-1, 1))

        self.u = np.array(((-2.0,), (2.0,)))

        """
        def covar_matrix(x, k):
            n = len(x)
            K = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    K[i,j] = k(x[i], x[j])
            return K
        k1 = lambda x1, x2 : 1.0*np.exp( - ((x1-x2)/1.5)**2 ) + (1.0 if x1==x2 else 0)
        K1 = covar_matrix(x, k1)

        np.random.seed(0)
        self.y1 = np.random.multivariate_normal(mean=np.zeros((len(x),)), cov=K1)
        """
        self.y1 = np.array([-1.02804007, -1.54448568, -0.31653812, -0.46768499, 0.67463927, 1.06519473, -1.39472442, -0.72392324, -2.99133689, -0.59922449, -3.70430871, -1.75810012, -0.80376896, -0.50514541, -0.5459166, 1.6353825, -1.13032502, 0.80372166, -0.01374143, -1.16083918, -1.6099601, -4.37523678, -1.53780366, -2.98047752, -3.41214803])




    def test_no_fic(self):
        # a CSFIC GP with a trivial FIC component should be equivalent to a plain GP.

        cov_main = GPCov(wfn_params=[1.0,], dfn_params=[ 2.5,], wfn_str="compact0", dfn_str="euclidean")
        cov_fic_tiny = GPCov(wfn_params=[0.00000000000001,], dfn_params=[ 0.0000000000001,], wfn_str="se", dfn_str="euclidean", Xu = self.u)
        noise_var = 1.0

        gp1 = GP(X=self.X,
                 y=self.y1,
                 noise_var = noise_var,
                 cov_main = cov_main,
                 cov_fic = cov_fic_tiny,
                 compute_ll=True,
                 sparse_threshold=0,
                 build_tree=False)

        gp2 = GP(X=self.X,
                 y=self.y1,
                 noise_var = noise_var,
                 cov_main = cov_main,
                 cov_fic = None,
                 compute_ll=True,
                 sparse_threshold=0,
                 build_tree=False)


        x_test = np.linspace(-6,6,20)
        pred1 = gp1.predict(np.reshape(x_test, (-1,1)))
        pred2 = gp2.predict(np.reshape(x_test, (-1,1)))

        self.assertTrue( ( np.abs(pred1-pred2) < 0.0001 ).all() )

        self.assertAlmostEqual(gp1.ll, gp2.ll)


    def test_no_cs(self):
        # a CSFIC GP with a trivial CS component should be equivalent to a plain FIC GP.

        # here we compare to "true" values from the GPStuff MATLAB toolbox on the same training data.
        # the code to reproduce these is in test/matlab/no_cs.m.

        cov_main_tiny = GPCov(wfn_params=[1e-15,], dfn_params=[ 1e-15,], wfn_str="compact0", dfn_str="euclidean")
        cov_fic = GPCov(wfn_params=[1.0,], dfn_params=[ 1.5,], wfn_str="se", dfn_str="euclidean", Xu = self.u)
        noise_var = 1.0

        gp = GP(X=self.X,
                 y=self.y1,
                 noise_var = noise_var,
                 cov_main = cov_main_tiny,
                 cov_fic = cov_fic,
                 compute_ll=True,
                 compute_grad=False,
                 sparse_threshold=0,
                 build_tree=False,
                sparse_invert=False)

        x_test = np.linspace(-6,6,20)
        pred = gp.predict(np.reshape(x_test, (-1,1)))

        true_pred = [-0.001009578312505, -0.007987369239529, -0.044328133303421, -0.172570712418792, -0.471267001236270, -0.902780793888510, -1.213226123858856, -1.144504376709686, -0.762372014152344, -0.378150072159795, -0.198456066738416, -0.207895507020449, -0.279193923665475, -0.292235035085667, -0.217163186457927, -0.113346671331517, -0.041505175187682, -0.010661391852200, -0.001921047707732, -0.000242814374795]
        self.assertTrue( ( np.abs(pred-true_pred) < 0.001 ).all() )


        true_ll = -52.1826173884063
        self.assertAlmostEqual(true_ll, gp.ll, places=2)

    def test_predict(self):
        cov_main = GPCov(wfn_params=[1.0,], dfn_params=[ 2.5,], wfn_str="compact0", dfn_str="euclidean")
        cov_fic = GPCov(wfn_params=[1.0,], dfn_params=[ 1.5,], wfn_str="se", dfn_str="euclidean", Xu = self.u)
        noise_var = 1.0

        gp = GP(X=self.X,
                 y=self.y1,
                 noise_var = noise_var,
                 cov_main = cov_main,
                 cov_fic = cov_fic,
                 compute_ll=True,
                 sparse_threshold=0,
                 build_tree=False)


        x_test = np.reshape(np.linspace(-6,6,20), (-1, 1))
        pred = gp.predict(x_test)

        true_pred = [-0.408347998350141, -0.562026829764370, -0.495194901764167, -0.221325067983013, -0.108486527530546, -0.382362268970377, -1.048794656436814, -1.603860074325163, -1.642537462422822, -1.301189117082765, -0.590641591880188,  0.013777393473535,  0.078987024190573, -0.060588451482957, -0.779405572439649, -1.566025186126343, -1.839795343563452, -1.918702553019862, -1.422525401522495, -0.783325324315960]
        self.assertTrue( ( np.abs(pred-true_pred) < 1e-7 ).all() )


        var = gp.variance(x_test)
        true_var = [1.845225336549096, 1.656486150287482, 1.467352884081546, 1.383325051904399, 1.238510769551563, 0.823748649278482, 0.390452316432541, 0.493252719187497, 0.988398813692138, 1.290702032271763, 1.290702032271763, 0.988398813692137, 0.493252719187498, 0.390452316432541, 0.823748649278483, 1.238510769551562, 1.383325051904399, 1.467352884081546, 1.656486150287482, 1.845225336549096]
        self.assertTrue( ( np.abs(var-true_var) < 1e-7 ).all() )

        true_ll = -45.986450666568985
        self.assertAlmostEqual(true_ll, gp.ll, places=8)


    def test_load_save(self):
        cov_main = GPCov(wfn_params=[1.0,], dfn_params=[ 2.5,], wfn_str="compact0", dfn_str="euclidean")
        cov_fic = GPCov(wfn_params=[1.0,], dfn_params=[ 1.5,], wfn_str="se", dfn_str="euclidean", Xu = self.u)
        noise_var = 1.0

        gp1 = GP(X=self.X,
                 y=self.y1,
                 noise_var = noise_var,
                 cov_main = cov_main,
                 cov_fic = cov_fic,
                 compute_ll=True,
                 sparse_threshold=0,
                 build_tree=False)


        gp1.save_trained_model("test_csfic.npz")
        gp2 = GP(fname="test_csfic.npz", build_tree=False)

        pts = np.reshape(np.linspace(-5, 5, 20), (-1, 1))
        p1 = gp1.predict(pts)
        v1 = gp1.variance(pts)
        p2 = gp2.predict(pts)
        v2 = gp2.variance(pts)

        self.assertTrue((p1 == p2).all())
        self.assertTrue((v1 == v2).all())

    def test_gradient(self):
        cov_main = GPCov(wfn_params=[.5,], dfn_params=[ 2.5,], wfn_str="compact2", dfn_str="euclidean")
        cov_fic = GPCov(wfn_params=[1.2,], dfn_params=[ 1.5,], wfn_str="se", dfn_str="euclidean", Xu = self.u)
        noise_var = 1.0

        gp = GP(X=self.X,
                 y=self.y1,
                 noise_var = noise_var,
                 cov_main = cov_main,
                 cov_fic = cov_fic,
                 compute_ll=True,
                 compute_grad=True,
                 compute_xu_grad=True,
                 sparse_threshold=0,
                 build_tree=False,
                sparse_invert=True)

        grad = gp.ll_grad

        nllgrad, x0, bounds, build_gp, _ = optimize_gp_hyperparams(X=self.X, y=self.y1, noise_var=noise_var, cov_main=cov_main, cov_fic=cov_fic, sparse_threshold=0)

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

        self.assertTrue( (np.abs(grad - empirical_grad) < 1e-6 ).all() )


    def test_sparse_gradient(self):
        cov_main = GPCov(wfn_params=[.5,], dfn_params=[ 2.5,], wfn_str="compact2", dfn_str="euclidean")
        cov_fic = GPCov(wfn_params=[1.2,], dfn_params=[ 1.5,], wfn_str="se", dfn_str="euclidean", Xu = self.u)
        noise_var = 1.0
        gp = GP(X=self.X,
                 y=self.y1,
                 noise_var = noise_var,
                 cov_main = cov_main,
                 cov_fic = cov_fic,
                 compute_ll=True,
                 compute_grad=True,
                 compute_xu_grad=True,
                 sparse_threshold=0,
                 build_tree=False,
                sparse_invert=True)


        g_sparse = gp._log_likelihood_gradient(self.y1, gp.Kinv)
        g_dense = gp._log_likelihood_gradient(self.y1, gp.Kinv.todense())
        self.assertTrue( (np.abs(g_sparse - g_dense) < 0.00001 ).all() )

class TestCSFICSemiParametric(unittest.TestCase):

    def setUp(self):
        N = 25
        x = np.linspace(-5,5,N)
        self.X = np.reshape(x, (-1, 1))

        self.u = np.array(((-2.0,), (2.0,)))

        self.basis = 'poly3'
        H, self.featurizer, self.featurizer_recovery = featurizer_from_string(self.X, self.basis, extract_dim=0)

        self.beta = np.array([9.91898792e-01,  -1.62113090e+00,   3.15437605e+00,   1.25732838e+00])*10
        self.B = np.eye(4) * 9
        self.b = np.zeros((4,))

        self.p = np.dot(H, self.beta)

        f1 = np.array([-1.02804007, -1.54448568, -0.31653812, -0.46768499, 0.67463927, 1.06519473, -1.39472442, -0.72392324, -2.99133689, -0.59922449, -3.70430871, -1.75810012, -0.80376896, -0.50514541, -0.5459166, 1.6353825, -1.13032502, 0.80372166, -0.01374143, -1.16083918, -1.6099601, -4.37523678, -1.53780366, -2.98047752, -3.41214803])
        self.y1 = f1 + self.p

        self.cov_main = GPCov(wfn_params=[0.3,], dfn_params=[ 0.5,], wfn_str="compact2", dfn_str="euclidean")
        self.cov_fic = GPCov(wfn_params=[0.2,], dfn_params=[ 2.5,], wfn_str="se", dfn_str="euclidean", Xu = self.u)
        self.noise_var = .001


        self.gp = GP(X=self.X,
                     y=self.y1,
                     noise_var = self.noise_var,
                     cov_main = self.cov_main,
                     cov_fic = self.cov_fic,
                     basis = self.basis,
                     featurizer_recovery = self.featurizer_recovery,
                     param_mean=self.b,
                     param_cov=self.B,
                     compute_ll=True,
                     compute_grad=True,
                     compute_xu_grad=True,
                     sparse_threshold=0,
                     build_tree=False,
                     sparse_invert=True)


    def test_load_save(self):

        gp1 = self.gp
        gp1.save_trained_model("test_semi_csfic.npz")
        gp2 = GP(fname="test_semi_csfic.npz", build_tree=False)

        pts = np.reshape(np.linspace(-5, 5, 20), (-1, 1))
        p1 = gp1.predict(pts)
        v1 = gp1.variance(pts)
        p2 = gp2.predict(pts)
        v2 = gp2.variance(pts)

        self.assertTrue((p1 == p2).all())
        self.assertTrue((v1 == v2).all())


    def test_gradient(self):
        grad = self.gp.ll_grad

        nllgrad, x0, bounds, build_gp, _ = optimize_gp_hyperparams(X=self.X, y=self.y1, basis=self.basis, featurizer_recovery=self.featurizer_recovery, param_mean=self.b, param_cov=self.B, noise_var=self.noise_var, cov_main=self.cov_main, cov_fic=self.cov_fic)

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



        self.assertTrue( (np.abs(grad - empirical_grad) < 0.001 ).all() )

    def test_sparse_gradient(self):
        g_sparse = self.gp._log_likelihood_gradient(self.y1, self.gp.Kinv)
        g_dense = self.gp._log_likelihood_gradient(self.y1, self.gp.Kinv.todense())
        self.assertTrue( (np.abs(g_sparse - g_dense) < 0.00001 ).all() )

    def test_limiting_csfic(self):
        gp_smallparam = GP(X=self.X,
                           y=self.y1,
                           noise_var = self.noise_var,
                           cov_main = self.cov_main,
                           cov_fic = self.cov_fic,
                           basis = self.basis,
                           featurizer_recovery = self.featurizer_recovery,
                           param_mean=np.zeros(self.b.shape),
                           param_cov=np.eye(len(self.b)) * 0.000000000000001,
                           compute_ll=True,
                           sparse_threshold=0,
                           build_tree=False,
                           sparse_invert=True)

        gp_noparam = GP(X=self.X,
                        y=self.y1,
                        noise_var = self.noise_var,
                        cov_main = self.cov_main,
                        cov_fic = self.cov_fic,
                        basis = None,
                        compute_ll=True,
                        sparse_threshold=0,
                        build_tree=False,
                        sparse_invert=True)


        # not sure why this doesn't pass
        self.assertGreater(self.gp.ll, gp_smallparam.ll)
        self.assertAlmostEqual(gp_smallparam.ll, gp_noparam.ll, places=5)

    def test_limiting_semiparametric(self):
        cov_fic_tiny = GPCov(wfn_params=[0.00000000000001,], dfn_params=[ 0.0000000000001,], wfn_str="compact2", dfn_str="euclidean", Xu = self.u)
        gp_smallfic = GP(X=self.X,
                           y=self.y1,
                           noise_var = self.noise_var,
                           cov_main = self.cov_main,
                           cov_fic = cov_fic_tiny,
                           basis = self.basis,
                           featurizer_recovery = self.featurizer_recovery,
                           param_mean=self.b,
                           param_cov=self.B,
                           compute_ll=True,
                           sparse_threshold=0,
                           build_tree=False,
                           sparse_invert=True)

        gp_nofic = GP(X=self.X,
                        y=self.y1,
                        noise_var = self.noise_var,
                        cov_main = self.cov_main,
                        cov_fic = None,
                        basis = self.basis,
                        featurizer_recovery = self.featurizer_recovery,
                        param_mean=self.b,
                        param_cov=self.B,
                        compute_ll=True,
                        sparse_threshold=0,
                        build_tree=False,
                        sparse_invert=True)

        self.assertAlmostEqual(gp_smallfic.ll, gp_nofic.ll, places=5)

if __name__ == '__main__':
    unittest.main()
