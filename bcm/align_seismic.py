
from sigvisa.utils.geog import dist_km
import numpy as np

def xcorr_valid(a,b):
    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(a)))

    xc = my_xc(a, b)
    xcmax = np.max(xc)
    offset = np.argmax(xc)
    return xcmax, offset, xc

import scipy.weave as weave
from scipy.weave import converters
def my_xc(a, b):
    # assume len(a) < len(b)
    n = len(b) - len(a)+1
    m = len(a)
    r = np.zeros((n,))
    a_normed = a / np.linalg.norm(a)
    code="""
for(int i=0; i < n; ++i) {
    double b_norm = 0;
    double cc = 0;
    for (int j=0; j < m; ++j) {
        cc += a_normed(j)*b(i+j);
        b_norm += b(i+j)*b(i+j);
    }
    cc /= sqrt(b_norm);
    r(i) = cc;
}
"""
    weave.inline(code,['n', 'm', 'a_normed', 'b', 'r',],type_converters = converters.blitz,verbose=2,compiler='gcc')
    """
    for i in range(n):
        window = b[i:i+len(a)]
        w_normed = window / np.linalg.norm(window)
        r[i] = np.dot(a_normed, w_normed)
    """
    return r



import cPickle as pickle

def load_events(sta="mkar"):
    s = []
    for i in range(1, 100):
        try:
            with open("/home/dmoore/p_waves/%s_stuff_%d" % (sta, i * 1000), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded", i
        except IOError:
            with open("/home/dmoore/p_waves/%s_stuff_final" % (sta,), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded final"
            break

    return s



s = load_events()


n = len(s)

window_start_idx = 20 # 2s before IDC arrival
window_end_idx = 260 # 8s after IDC arrival (so, 10s window)

xcmax1 = np.memmap("xcmax1", dtype='float32', mode='w+', shape=(n,n))
xcmax2 = np.memmap("xcmax2", dtype='float32', mode='w+', shape=(n,n))
xcmax = np.memmap("xcmax", dtype='float32', mode='w+', shape=(n,n))

offset1 = np.memmap("offset1", dtype='int32', mode='w+', shape=(n,n))
offset2 = np.memmap("offset2", dtype='int32', mode='w+', shape=(n,n))

t = np.linspace(-3.0, 10.0, 261)
prior = -t/3.0

distances = np.zeros((n, n))

for i, (ev1, (w1, srate1)) in enumerate(s[:n]):
    patch1 = w1[window_start_idx:window_end_idx]
    for j, (ev2, (w2, srate2)) in enumerate(s[:i]):
        patch2 = w2[window_start_idx:window_end_idx]

        xc1 = my_xc(patch1, w2)
        xc2 = my_xc(patch2, w1)

        align1 = np.argmax(xc1 + prior)
        align2 = np.argmax(xc2 + prior)

        offset1[i,j] = align1
        offset2[i,j] = align2

        xcmax1[i,j] = xc1[align1]
        xcmax2[i,j] = xc2[align2]
        #xcmax1[j,i] = xc1[align1]
        #xcmax2[j,i] = xc2[align2]
        xcmax[i,j] = max(xcmax1[i,j], xcmax2[i,j])
        #xcmax[j,i] = max(xcmax1[j,i], xcmax2[j,i])

        #ll_dist = dist_km((ev1.lon, ev1.lat), (ev2.lon, ev2.lat))
        #depth_dist = np.abs(ev1.depth-ev2.depth)
        #dist = np.sqrt(ll_dist**2 + depth_dist**2)
        #distances[i,j] = dist
        #distances[j,i] = dist
    print "correlated", i

del xcmax1
del xcmax2
del xcmax
del offset1
del offset2
