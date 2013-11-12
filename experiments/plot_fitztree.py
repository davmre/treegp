from sigvisa.plotting.event_heatmap import EventHeatmap
import numpy as np
import sys

from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg



def main(treedump_fname, out_fname):
    f = open(treedump_fname, 'r')
    pts = []
    for line in f:
        lon,lat,r, d = line.split(" ")
        pts.append( (float(lon), float(lat), float(r), int(d)) )
    mypts = np.asarray(pts)
    pts = mypts#[0:5999:10, :]

    f = Figure((11,8))
    ax = f.add_subplot(111)

    hm = EventHeatmap(f=None, autobounds=pts, autobounds_quantile=.9995, calc=False, )
    normed_locations = [hm.normalize_lonlat(*location) for location in pts[:, 0:2]]

    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)
    #hm.add_stations(("FITZ",))
    #fitz_location = [hm.sitenames[n][0:2] for n in hm.stations]
    #hm.plot_locations(fitz_location, labels=None,
    #                  marker="x", ms=10, mfc="none", mec="blue", mew=4, alpha=1)


    for enum, ev in enumerate(normed_locations):
        x, y = hm.bmap(ev[0], ev[1])
        radius = pts[enum, 2]/111.0
        depth = pts[enum, 3]

        if radius == 0:
            hm.bmap.plot([x], [y], zorder=1, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
            pass
        else:
            if depth < 10:
                alpha = min(1.0, 5.0/(depth**2+1.0))

                ka = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5, 'alpha': alpha}
                ax.add_patch(Circle((x,y), radius, **ka))


    #hm.plot_locations(pts[:, 0:2], labels=None,
    #

    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(out_fname, bbox_inches="tight")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        treedump_fname = "papers/product_tree/fitztree.txt"
        out_fname = "papers/product_tree/fitz_ctree.png"
    else:
        treedump_fname = sys.argv[1]
        out_fname = sys.argv[2]
    main(treedump_fname, out_fname)
