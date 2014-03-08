from sigvisa.plotting.event_heatmap import EventHeatmap
import numpy as np
import scipy.stats

import os

from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm

from sigvisa import Sigvisa
from sigvisa.models.spatial_regression.SparseGP import SparseGP
from sigvisa.learn.train_param_common import subsample_data

from synth_dataset import mkdir_p, eval_gp
from real_dataset import train_realdata_gp, test_predict, learn_hyperparams

def plot_matrix(M, filename):

    #def add_alpha(M_img):
    #    M_img[:, :, 3] = ( (3.9215686 - np.sum(M_img, axis=2)) > 0 ) * 1.0


    M = M / scipy.stats.scoreatpercentile(M, 99.9)
    M_img = cm.Purples(M)
    #add_alpha(M_img)

    f = Figure((11,8))
    ax = f.add_subplot(111)
    ax.imshow(M_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(filename, bbox_inches="tight", dpi=300, transparent=True)


def plot_kernel_matrix(sgp):
    Kinv = np.power(np.abs(sgp.Kinv.todense()), 1.0/3)
    plot_matrix(Kinv, "fitz_X_Kinv.png")

    K = np.power(np.abs(sgp.K.todense()), 1.0/3)
    plot_matrix(K, "fitz_X_K.png")

    print 1.0 - float(np.sum(K > 0))/ np.sum(K >= 0)
    print 1.0 - float(np.sum(Kinv > 0))/ np.sum(Kinv >= 0)

    #Kinv_sparsity = (Kinv > 0) * 1.0
    #plot_matrix(Kinv_sparsity, "fitz_X_Kinv_sparsity.png")

    #K_sparsity = (K > 0) * 1.0
    #plot_matrix(K_sparsity, "fitz_X_K_sparsity.png")


def plot_events_and_GP(X, y, sgp):
    Xorig = X
    X = np.array(X[:, 0:2])

    s = Sigvisa()
    fitz_location = [s.earthmodel.site_info("FITZ", 0)[0:2],]
    print fitz_location
    f = Figure((11,8))
    ax = f.add_subplot(111)
    fmean = lambda x, y: sgp.predict(np.array(((x, y, 0.0),)))
    fvar = lambda x, y: np.sqrt(sgp.variance(np.array(((x, y, 0.0),))))
    hm = EventHeatmap(f=fmean, autobounds=X, autobounds_quantile=0.96, calc=True, n=200)
    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)
    hm.plot_locations(fitz_location, labels=None,
                      marker="x", ms=10, mfc="none", mec="blue", mew=4, alpha=1)

    #hm.plot_locations(X, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
    scplot = hm.plot_locations(X, marker=".", s=40, edgecolors="none", alpha=.5, yvals=y, yval_colorbar=True)
    ax.set_frame_on(False)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig("fitz_X.png", bbox_inches="tight", dpi=300, transparent=True, )


    hm.plot_density(smooth=True, colorbar=False, nolines=True, cm=scplot.get_cmap(), vmin=0.0, vmax=2.4)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig("fitz_X_gp.png", bbox_inches="tight", dpi=300, transparent=True)


    f = Figure((11,8))
    ax = f.add_subplot(111)

    hm = EventHeatmap(f=fvar, autobounds=X, autobounds_quantile=0.96, calc=True, n=200)
    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)
    hm.plot_locations(fitz_location, labels=None,
                      marker="x", ms=10, mfc="none", mec="blue", mew=4, alpha=1)

    #hm.plot_locations(X, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
    ax.set_frame_on(False)
    hm.plot_density(smooth=True, colorbar=True, nolines=True)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig("fitz_X_gp_std.png", bbox_inches="tight", dpi=300, transparent=True)


def plot_tree(fname, include_points=True, max_depth=9999):
    X = np.loadtxt("fitzX.txt")
    f = open("fitztree.txt", 'r')
    pts = []
    for line in f:
        lon,lat,r, d = line.split(" ")
        if int(d) < max_depth:
            pts.append( (float(lon), float(lat), float(r), int(d)) )
    mypts = np.asarray(pts)
    pts = mypts#[0:5999:10, :]

    f = Figure((11,8))
    ax = f.add_subplot(111)

    hm = EventHeatmap(f=None, autobounds=X, autobounds_quantile=0.96, calc=False, )
    normed_locations = [hm.normalize_lonlat(*location) for location in pts[:, 0:2]]

    hm.init_bmap(axes=ax, nofillcontinents=not include_points, coastlines=True)
    if include_points:
        hm.plot_earth(y_fontsize=16, x_fontsize=16)
        hm.add_stations(("FITZ",))
        fitz_location = [Sigvisa().earthmodel.site_info("FITZ", 0)[0:2],]
        hm.plot_locations(fitz_location, labels=None,
                          marker="x", ms=10, mfc="none", mec="blue", mew=4, alpha=1)


    for enum, ev in enumerate(normed_locations):
        x, y = hm.bmap(ev[0], ev[1])
        radius = pts[enum, 2]/111.0
        depth = pts[enum, 3]

        if radius == 0 and include_points:
            hm.bmap.plot([x], [y], zorder=1, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
        else:
            alpha = 1.0/(np.sqrt(np.sqrt(depth)+1.0))
            ka = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1, 'alpha': alpha}
            ax.add_patch(Circle((x,y), radius, **ka))



    ax.set_frame_on(False)
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(fname, bbox_inches="tight", dpi=300, transparent=True)


def learn_gp(X, y):

    p = np.random.permutation(len(y))
    train_n = int(len(y) * 0.2)
    trainX = X[p[:train_n], :]
    trainy = y[p[:train_n]]
    testX = X[p[train_n:], :]
    testy = y[p[train_n:]]

    fitz_dir = os.path.join(os.getenv("SIGVISA_HOME"), "papers", "product_tree", "fitz_learned")

    #hyperparams = np.array([0.5,  3.0, 50.0,  50.0], dtype=float)
    #hyperparams = learn_hyperparams(fitz_dir, trainX, trainy, dfn_str='lld', hyperparams=hyperparams, sparse_invert=False, basisfns = [lambda x : 1,], param_cov=np.array(((10000,),)), param_mean = np.array((0,)), k=1000)

    #print "got hyperparams", hyperparams

    #hyperparams = np.array([1.16700753,    2.53145332,  212.46536884,157.68719303], dtype=float)

    np.save(os.path.join(fitz_dir, "testX.npy"), testX)
    np.save(os.path.join(fitz_dir, "testy.npy"), testy)
    np.save(os.path.join(fitz_dir, "hyperparams.npy"), hyperparams)

    print "loaded data"

    train_realdata_gp(fitz_dir, trainX, trainy, hyperparams=hyperparams, sparse_invert=False, basisfns = [lambda x : 1,], param_cov=np.array(((10000,),)), param_mean = np.array((0,)), dfn_str='lld')
    test_predict(fitz_dir)
    eval_gp(bdir=fitz_dir, test_n=100)


def main():

    X = np.loadtxt("fitzX.txt")
    y = np.loadtxt("fitzy.txt")

    learn_gp(X, y)
    return

    #X, y = subsample_data(X, y, k=1000)

    sgp = SparseGP(X=X, y=y, hyperparams=[.5, 3, 50, 50], basisfns = [lambda x : 1,], param_cov=np.array(((10000,),)), param_mean = np.array((0,)), sparse_invert=True, sort_events=True)

    plot_events_and_GP(X, y, sgp)

if __name__ == "__main__":
    try:
        #for d in (1, 2, 3, 4, 6, 8, 10, 13, 15, 20):
        #    plot_tree("fitz_ctree_%03d.png" % d, max_depth=d, include_points=False)
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import sys, traceback, pdb
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
