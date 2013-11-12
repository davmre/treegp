import numpy as np

from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sigvisa.plotting.heatmap import Heatmap
from scipy.spatial import KDTree

f = open("papers/product_tree/tco.csv", 'r')
data_dict = {}
lats = set()
lons = set()
for line in f:
    lon, lat, depth, tco = [float(x) for x in line.split(',')]
    lats.add(lat)
    lons.add(lon)
    data_dict[(lon, lat)] = tco

lats = np.array(sorted(list(lats)))
lons = np.array(sorted(list(lons)))

def neighbor(lon, lat):
    idx1 = np.searchsorted(lons, lon)
    idx2 = np.searchsorted(lats, lat)

    try:
        lon2 = lons[idx1-1]
    except IndexError:
        lon2 = lons[idx1]
    try:
        if np.abs(lons[idx1] - lon) < np.abs(lon2-lon):
            lon2 = lons[idx1]
    except IndexError:
        pass

    try:
        lat2 = lats[idx2-1]
    except IndexError:
        lat2 = lats[idx2]
    try:
        if np.abs(lats[idx2] - lat) < np.abs(lat2-lat):
            lat2 = lats[idx2]
    except IndexError:
        pass

    return lon2, lat2

def lookup(lon, lat):
    lon2,lat2 = neighbor(lon, lat)
    tco = data_dict[(lon2, lat2)]
    return tco

hm = Heatmap(f = lookup, left_lon=-179, right_lon=179, top_lat=89.5, bottom_lat=-89.5, n=180)
fig = Figure(figsize=(8, 5), dpi=144)
fig.patch.set_facecolor('white')
axes = fig.add_subplot(111)
hm.init_bmap(axes=axes)
hm.plot_density(smooth=True, nolines=True)
canvas = FigureCanvasAgg(fig)
canvas.print_figure("tco.png")
