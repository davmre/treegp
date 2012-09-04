import numpy as np
 import matplotlib as plt

def predict_1d (
    gp,
    num_steps = 100,
    x_min = None,
    x_max = None,
    ):
    """
    Plot a gp's prediction with error bars of 2*std.
    """
    if None == x_min: x_min = min( x[0] for x in gp.X )
    if None == x_max: x_max = max( x[0] for x in gp.X )
    x_max = float( x_max )
    x_min = float( x_min )
    predict_x = np.reshape(np.linspace( x_min, x_max, num_steps ), (-1, 1))

    mean = gp.predict( predict_x )
    variance = gp.variance( predict_x )
    plt.figure()

    data = [
        (x,y,max(v,0.0))
        for x,y,v
        in zip( predict_x, mean.flat, variance  )
        ]
    data.sort( key = lambda d: d[0] ) # sort on X axis

    # plot mean predictions
    predict_x = [ d[0] for d in data ]
    predict_y = np.array( [ d[1] for d in data ] )
    plt.plot( predict_x, predict_y, color='k', linestyle=':' )

    # plot error bars
    sd = np.sqrt( np.array( [ d[2] for d in data ] ) )
    var_x = np.concatenate((predict_x, predict_x[::-1]))
    var_y = np.concatenate((predict_y + 2.0 * sd, (predict_y - 2.0 * sd)[::-1]))
    p = plt.fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3')

def interpolate_surface(gp, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xmin = np.min(X, 0)
    xmax = np.max(X, 0)

    u = np.linspace(xmin[0], xmax[0], 20)
    v = np.linspace(xmin[1], xmax[1], 20)

    xc = np.outer(np.ones((20,)), u)
    yc = np.outer(v, np.ones((20,)))

    k = np.zeros(xc.shape)
    for i in range(xc.shape[0]):
        for j in range(xc.shape[1]):
            k[i,j] = gp.predict((xc[i,j], yc[i,j]))

    #print xmin, xmax
    #print u, v
    #print x, y, k

    ax.plot_surface(xc, yc, k,  color='b')

    plt.show()
