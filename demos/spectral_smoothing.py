from dolfin import *
import numpy as np

from scipy.fft import rfft, irfft
from slash.convex_hull import convex_hull
from bastian_inside_folds import walk_vertex_distance_from, walk_vertex_distance

import matplotlib.pyplot as plt

bdry_mesh = Mesh('clipped_2_mesh.xml')
X = bdry_mesh.coordinates()

cvx_hull = convex_hull(X)
idx, dist_h, hull_idx = np.array([
    p for p in walk_vertex_distance_from(bdry_mesh, cvx_hull, True)
]).reshape((-1, 3)).T
idx, hull_idx = np.asarray(idx, 'uintp'), np.asarray(hull_idx, 'uintp')
# The idea of this smoothing process is to work with arc-length coordinates
# of the surface. Using convexhull we can for each point define a distance 
# to the hull and obtain a signal distance vs arcl on which we can do FFT
_, arcl = np.fromiter(sum(walk_vertex_distance(bdry_mesh), ()), dtype=float).reshape((-1, 2)).T

plt.figure()
N = len(arcl-1)

signal = dist_h
transf = rfft(signal, N)
# Now we kick out some frequencies whose intensity is low
transf_large = np.where(np.abs(transf) < 5E2, np.zeros(len(transf)), transf)
# build a smoother signal
filtered = irfft(transf_large, N)

Y = []
# The challenge now is to go back. Here we do it by "walking" shorter/filtered
# distance on the segment that was used originally to define the hull distance
for i, hull_i, d in zip(idx, hull_idx, filtered):
    xi, yi = X[i], cvx_hull[hull_i]
    n = (xi-yi)/np.linalg.norm(yi-xi)

    Y.append(yi + d*n)
Y = np.array(Y)
#
# The problem is that this does not really smooth the surface. In particular
# it can entagle the edges!
#
plt.figure()
plt.plot(arcl, signal)
plt.plot(arcl, filtered)
plt.show()

plt.figure()
plt.plot(X[idx, 0], X[idx, 1])
plt.plot(Y[:, 0], Y[:, 1])
plt.axis('equal')
plt.show()
