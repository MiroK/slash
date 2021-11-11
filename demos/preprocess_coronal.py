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


def smooth(contour_mesh, temperature, alpha=0.2, prefix='', forced=True):
    '''By heat equation'''
    idx, arcl = np.fromiter(sum(walk_vertex_distance(contour_mesh), ()), dtype=float).reshape((-1, 2)).T
    idx = idx.astype('uintp')
        
    inside_contour = contour_mesh.coordinates()[idx]

    # Encloses this shape in a frame to get a geometry on which we
    #  will do the smoothing
    ll, ur = inside_contour.min(axis=0), inside_contour.max(axis=0)
    du = ur - ll
    ll -= 0.125*du
    ur += 0.125*du

    bbox = np.array([ll,
                     [ur[0], ll[1]],
                     ur,
                     [ll[0], ur[1]],
                     ll])

    # For API purposes we want one mesh
    mesh, cell_f = ContourMesh([bbox, inside_contour])

    # Now the contours are used to define a mesh on which we want to solve
    # heat equation for smoothing
    heat_mesh_path, heat_bdry_path = '_'.join([prefix, 'heat_eq_mesh.xml']), '_'.join([prefix, 'heat_eq_bdries.xml'])
    # Generate
    if forced or not all(os.path.exists(f) for f in (heat_mesh_path, heat_bdry_path)):
        _, entity_fs = mesh_bounded_surface(cell_f, scale=1/2.**4, view=False, loop_opts={1: False, 2: True})
        bdries = entity_fs[1]
        heat_mesh = bdries.mesh()

        df.File(heat_mesh_path) << heat_mesh
        df.File(heat_bdry_path) << bdries
    # Load
    else:
        heat_mesh = df.Mesh(heat_mesh_path)
        bdries = df.MeshFunction('size_t', heat_mesh, heat_bdry_path)

    # Now we smooth
    dirichlet_data = {2: df.Constant(1)}
    neumann_data = {1: df.Constant(0)}
    u0 = df.Constant(0)
    f = df.Constant(0)

    for t, u0 in solve_heat(alpha, bdries, f, u0, dirichlet_data, neumann_data):
        # NOTE: maybe you want to store at intermediate times 
        continue

    # Check the smoothing effect
    plt.figure()
    # Original
    plt.plot(inside_contour[:, 0], inside_contour[:, 1])
    # Smoothed
    cvalues = [temperature]

    X, pieces = get_contourlines(u0, cvalues)
    for cval, pieces_of_contour in pieces:
        line = None
        # This is how we'd look at all the pieces
        # for k, contour_piece in enumerate(pieces_of_contour, 1):
        #     x, y = X[contour_piece].T
        #     line, = plt.plot(x, y, color=line.get_color() if line is not None else None)
        # print(f'{cval} has {k} pieces')

        # But we probably want only the largest one
        clen = lambda pts, X=X: contour_len(X, pts)
        contour_piece = max(pieces_of_contour, key=clen)
        x, y = X[contour_piece].T
        line, = plt.plot(x, y, color=line.get_color() if line is not None else None)
    plt.axis('equal')
    plt.show()

    return ContourMesh(X[contour_piece])

# --------------------------------------------------------------------

if __name__ == '__main__':
    from contour_meshing import mesh_bounded_contours
    
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), "/home/mirok/Downloads/021_coronal_16/brain_mesh.h5", "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    hdf.read(subdomains, "/subdomains")

    cc, cc_bdries, lookup, _ = mark_connected_components(subdomains)

    df.File('cc.pvd') << cc
    df.File('cc_bdries.pvd') << cc_bdries

    cc, cc_bdries, lookup = close_small_volumes(tol=10, cell_f=cc, facet_f=cc_bdries, lookup=lookup)    
    # cc, cc_bdries, lookup = close_small_volumes(tol=100, cell_f=cc, facet_f=cc_bdries, lookup=lookup)

    df.File('closed_cc.pvd') << cc
    df.File('closed_cc_bdries.pvd') << cc_bdries

    which = 'noHole'

    # The one that needs to be smoothed
    inner_contour = (2, )
    inner_contour_mesh = EmbeddedMesh(cc_bdries, inner_contour)
    inner_contour_mesh, _ = smooth(inner_contour_mesh, temperature=0.75, prefix='inner', forced=False)

    outer_contour = (1, 3, 4, 5, 6, 7)
    outer_contour_mesh = EmbeddedMesh(cc_bdries, outer_contour)
    cell_f, branch_colors, loop_colors = color_branches(outer_contour_mesh)

    outer_contour_mesh = EmbeddedMesh(cell_f, tuple(set(branch_colors) - set(loop_colors)))

    # We have a few more
    c_8 = EmbeddedMesh(cc_bdries, 8)
    c_9 = EmbeddedMesh(cc_bdries, 9)
    c_11 = EmbeddedMesh(cc_bdries, 11)
    c_12 = EmbeddedMesh(cc_bdries, 12)
    c_18 = EmbeddedMesh(cc_bdries, 18)

    # Without hole
    if which == 'noHole':
        surfaces = [(1, 2), (2, )]
        contours = {1: outer_contour_mesh, 2: inner_contour_mesh}
        
    elif which == 'hole':
        # With holes
        surfaces = {(1, 2), (2, 8, 9, 11, 12, 18)}
        contours = {1: outer_contour_mesh,
                    2: inner_contour_mesh,
                    8: c_8, 9: c_9, 11: c_11, 12: c_12, 18: c_18}

    elif which == 'holeFilled':
        # With holes
        surfaces = {(1, 2),
                    (2, 8, 9, 11, 12, 18),
                    (8, ), (9, ), (11, ), (12, ), (18, )}
        contours = {1: outer_contour_mesh,
                    2: inner_contour_mesh,
                    8: c_8, 9: c_9, 11: c_11, 12: c_12, 18: c_18}
        

    for i in range(2, 7):
        smesh, entity_fs = mesh_bounded_contours(contours, surfaces, scale=1/2**i, view=False)
        print(smesh.hmin(), sum(entity_fs[1].array() == 2), '<<')
        
        with df.HDF5File(df.MPI.comm_world, f'{which}_brain_skull_coloronal{i}.h5', 'w') as out:
            out.write(smesh, 'mesh')
            out.write(entity_fs[2], 'volumes')
            out.write(entity_fs[1], 'surfaces')

    df.File(f'{which}_brain_skull_coronal.pvd') << smesh
    df.File(f'{which}_brain_skull_coronal_subdomains.pvd') << entity_fs[2]    
    df.File(f'{which}_brain_skull_coronal_interfaces.pvd') << entity_fs[1]
