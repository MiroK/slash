from bastian_inside_folds import mesh_bounded_surface, walk_vertex_distance
from contour_smoothing import solve_heat, get_contourlines, contour_len
from bastian_inside_folds import mesh_bounded_surface

from slash.cc_coloring import close_small_volumes, mark_connected_components
from slash.embedded_mesh import ContourMesh, EmbeddedMesh
from slash.branching import color_branches

import matplotlib.pyplot as plt
import dolfin as df
import numpy as np
import os

from preprocess_coronal import smooth
from slash.make_mesh_cpp import make_mesh


def line_mesh(points, is_loop=True):
    '''Made of sequences of points'''
    tdim = 1
    nvtx, gdim = points.shape

    if is_loop:
        cells = np.array([(i, (i+1)%nvtx) for i in range(nvtx)])
    else:
        cells = np.array([(i, i+1) for i in range(nvtx-1)])
    
    mesh = df.Mesh()
    make_mesh(coordinates=points, cells=cells, tdim=tdim, gdim=gdim,
              mesh=mesh)

    return mesh


def perturbed_circle(center, radius, amplitude, frequency, npts=300):
    '''Put a sin wave in circle'''
    x0, y0 = center

    thetas = 2*np.pi*np.linspace(0, 1, npts)
    # The circle
    x = x0 + radius*np.cos(thetas)
    y = y0 + radius*np.sin(thetas)

    if amplitude == 0 or frequency == 0:
        X, Y = x, y
    else:
        # Tangent
        dx = -radius*np.sin(thetas)
        dy = radius*np.cos(thetas)
        
        dl = np.sqrt(dx**2 + dy**2)
        # Unit normal
        u = dy/dl
        v = -dx/dl

        f = lambda th: amplitude*np.sin(frequency*th)

        X = x + u*f(thetas)
        Y = y + v*f(thetas)

    return np.c_[X, Y][:-1]

# --------------------------------------------------------------------

if __name__ == '__main__':
    from contour_meshing import mesh_bounded_contours

    synthetic_brain = {
        'radius_scale': 2.0,
        'amplitude': 2,   # Of the surface oscillations
        'frequency': 8,  # 2*pi*frequency is the oscillation freq
    }
    
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), "/home/mirok/Downloads/021_coronal_16/brain_mesh.h5", "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    hdf.read(subdomains, "/subdomains")

    cc, cc_bdries, lookup, _ = mark_connected_components(subdomains)

    cc, cc_bdries, lookup = close_small_volumes(tol=10, cell_f=cc, facet_f=cc_bdries, lookup=lookup)    

    # The one that needs to be smoothed
    inner_contour = (2, )
    inner_contour_mesh = EmbeddedMesh(cc_bdries, inner_contour)
    inner_contour_mesh, _ = smooth(inner_contour_mesh, temperature=0.75, prefix='inner', forced=False)

    if synthetic_brain:
        # Want to replace synthetic with a circular hole
        coords = inner_contour_mesh.coordinates()
        center = coords.mean(axis=0)

        radius = np.min(np.linalg.norm(coords-center, 2, axis=1))

        radius = radius*synthetic_brain['radius_scale']
        amplitude, frequency = synthetic_brain['amplitude'], synthetic_brain['frequency']
        
        points = perturbed_circle(center, radius, amplitude, frequency, npts=300)        

        inner_contour_mesh = line_mesh(points)
        df.File('foo.pvd') << inner_contour_mesh
    
    outer_contour = (1, 3, 4, 5, 6, 7)
    outer_contour_mesh = EmbeddedMesh(cc_bdries, outer_contour)
    cell_f, branch_colors, loop_colors = color_branches(outer_contour_mesh)

    outer_contour_mesh = EmbeddedMesh(cell_f, tuple(set(branch_colors) - set(loop_colors)))

    surfaces = [(1, 2), (2, )]
    contours = {1: outer_contour_mesh, 2: inner_contour_mesh}

    # Different refiments
    for i in range(2, 5):
        smesh, entity_fs = mesh_bounded_contours(contours, surfaces, scale=1/2**i, view=False)
        print(smesh.hmin(), sum(entity_fs[1].array() == 2), '<<')
        
        with df.HDF5File(df.MPI.comm_world, f'synthetic_brain_{i}.h5', 'w') as out:
            out.write(smesh, 'mesh')
            out.write(entity_fs[2], 'volumes')
            out.write(entity_fs[1], 'surfaces')

    freq = synthetic_brain.get('frequency', 0)
            
    df.File(f'synthetic_brain_subdomains_{freq}.pvd') << entity_fs[2]    
    df.File(f'synthetic_brain_interfaces_{freq}.pvd') << entity_fs[1]
