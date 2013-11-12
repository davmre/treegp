import numpy as np
import time

from sigvisa.models.spatial_regression.SparseGP_CSFIC import SparseGP_CSFIC

def main():
    X_train = np.loadtxt('X_train.txt', delimiter=',')
    y_train = np.loadtxt('y_train.txt', delimiter=',')
    X_test = np.loadtxt('X_test.txt', delimiter=',')
    y_test = np.loadtxt('y_test.txt', delimiter=',')

    #X_train = np.array(X_train[0:300, :])
    #y_train = np.array(y_train[0:300])

    Xu = np.loadtxt('Xu.txt', delimiter=',')

    dfn_str = "euclidean"
    dfn_params_fic =  np.array([3.5328, 7.3501], dtype=float)
    dfn_params_cs =  np.array([2.0638, 2.6316], dtype=float)

    wfn_str_fic = "se"
    wfn_params_fic = np.array([16.6132,], dtype=float)

    wfn_str_cs = "compact2"
    wfn_params_cs = np.array([4.5806,], dtype=float)

    noise_var = 3.5349

    """
    gp1 = SparseGP_CSFIC(X=X_train, y=y_train,
                        dfn_str=dfn_str, dfn_params_fic=dfn_params_fic, dfn_params_cs=dfn_params_cs,
                        wfn_str_fic = wfn_str_fic, wfn_params_fic=wfn_params_fic,
                        wfn_str_cs = wfn_str_cs, wfn_params_cs=wfn_params_cs,
                        build_tree=True, sort_events=False, center_mean=False,
                        noise_var=noise_var, Xu=Xu)

    gp1.save_trained_model('trained_gp.gp')
    """
    gp = SparseGP_CSFIC(fname='trained_gp.gp', build_tree=True)

    t0 = time.time()
    predictions = gp.predict_naive(X_test)
    t1 = time.time()
    prediction_tree = gp.predict(X_test)
    t2 = time.time()

    print "mad", np.mean(np.abs(predictions - y_test)), (t1-t0)
    print "mad tree", np.mean(np.abs(prediction_tree - y_test)), (t2-t1)

    t3 = time.time()
    lps = [gp.log_p(x=y, cond=np.reshape(x, (1,-1))) for (x,y) in zip(X_test, y_test)]
    t4 = time.time()
    lps_tree = [gp.log_p(x=y, cond=np.reshape(x, (1,-1)), covar="double_tree", covar_args={'eps': 1e-4}) for (x,y) in zip(X_test, y_test)]
    t5 = time.time()
    lps_naive = [gp.log_p(x=y, cond=np.reshape(x, (1,-1)), covar="naive") for (x,y) in zip(X_test, y_test)]
    t6 = time.time()

    print "mlpd", np.mean(lps), (t4-t3)
    print "mlpd tree", np.mean(lps_tree), (t5-t4)
    print "mlpd naive", np.mean(lps_naive), (t6-t5)

    import pdb; pdb.set_trace()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import sys, traceback, pdb
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
