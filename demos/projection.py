from slash.convex_hull import convex_hull
import numpy as np


def slice_projection(x, x0, n):

    n = n/np.linalg.norm(n)

    z = x - x0
    y = z - n*(np.sum(z*n, axis=1)).reshape((-1, 1))
    # Want to express the prejected points now in the plane coordinates
    eigw, eigv = np.linalg.eigh(np.outer(n, n))
    eigv = eigv.T

    u, v = eigv[np.abs(eigw) < 1E-10]

    u = np.sum(z*u, axis=1)
    v = np.sum(z*v, axis=1)

    return convex_hull(np.c_[u, v])


# ----------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import UnitCubeMesh
    import matplotlib.pyplot as plt

    mesh = UnitCubeMesh(32, 32, 32)

    x0 = np.array([0.4, 0.2, 0.5])
    n = np.array([0.2, 0.4, 0.3])

    x = mesh.coordinates()
    contour = slice_projection(x, x0, n)
    contour = np.row_stack([contour, contour[0]])
    
    plt.figure()
    plt.plot(contour[:, 0], contour[:, 1])
    plt.axis('equal')
    plt.show()
    
