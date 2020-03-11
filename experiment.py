import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D



archive = np.load("/home/filip/Documents/Potential3D/VideoPose3D-master/keypoints3D.npy")
for keypoints in archive:
    fig = pyplot.figure()
    ax = Axes3D(fig)
    x, y, z = [[element[0] for element in keypoints], [element[1] for element in keypoints],
               [element[2] for element in keypoints]]
    ax.scatter(x, y, z)
    pyplot.show()
