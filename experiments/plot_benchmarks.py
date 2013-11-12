from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np

f = open('papers/product_tree/benchmarks.txt', 'r')
data = []
f.readline()
for line in f:
    data.append([float(k) for k in line.split('\t')])
data= np.array(data)
npts = data[:,0]
sparse = data[:,4]*1000
spkernel = data[:,5] * 1000
tree8 = data[:,6] * 1000


fig = Figure()
ax = fig.add_subplot(1,1,1)
ax.plot(npts/1000, sparse, marker='.', linestyle='--', c='black', label="Direct")
ax.plot(npts/1000, spkernel, marker='.', linestyle='-', c='black', label="Hybrid")
ax.plot(npts/1000, tree8, marker='.', linestyle=':', c='black', label="ProductTree")
ax.set_ylim((0, 10))
ax.set_xlim((1, 170))
ax.set_xlabel("$n$ (thousands)", fontsize=20)
ax.set_ylabel("time (ms)", fontsize=20)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

canvas = FigureCanvasAgg(fig)
canvas.draw()
fig.savefig("papers/product_tree/synthetic.png")
