from slash.gmsh_interopt import msh_gmsh_model, mesh_from_gmsh
from slash.branching import color_branches, is_loop, walk_vertices
from slash.embedded_mesh import ContourMesh
from slash.utils import *

import dolfin as df
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import gmsh, sys


def walk_vertex_distance(mesh):
    '''Parametrize the 1d surface by distance from first point'''
    assert is_loop(mesh)
    
    x = mesh.coordinates()

    vertices = walk_vertices(mesh)

    d, v0 = 0, next(vertices)

    yield v0, d
    
    for v in vertices:
        d += np.linalg.norm(x[v] - x[v0])
        yield v, d
        v0 = v

        
def walk_vertex_distance_from(mesh, shape_points, return_index=False):
    '''Walk along while computeting distance from shape (represented by points)'''
    assert is_loop(mesh)
    assert shape_points.ndim == 2
    
    x = mesh.coordinates()

    for v in walk_vertices(mesh):
        i = np.argmin(np.linalg.norm(shape_points - x[v], 2, axis=1))
        if return_index:
            yield v, np.linalg.norm(shape_points[i] - x[v]), i
        else:
            yield v, np.linalg.norm(shape_points[i] - x[v])


def mesh_bounded_surface(mesh1d, scale=1., view=False):
    '''Mesh the inside bounded by surfaces'''
    # FIXME: lots of assumptions here on how the surfaces are enclosing each other
    center = np.mean(mesh1d.coordinates(), axis=0).reshape((1, 2))
    # ... and find the bounding surfaces
    closed_surfaces, tags = connected_domains(mesh1d)

    # Some of these loops practically enclose no volume
    valid_bdries = []
    for tag in range(*tags):
        surf_mesh = entity_mesh(closed_surfaces, tag)
        vol = approx_enclosed_volume(surf_mesh)
        # Remove "empty" clusters
        vol > 1E-10 and valid_bdries.append(surf_mesh)

    loops = []
    # We restrict ourselves here to bounding surfaces that are simple
    # loops, i.e, there is no branching for they are easy to handle for gmsh
    loop_len = lambda t, m: volume_function(m, t).vector().sum()

    for tag, surf_mesh in enumerate(valid_bdries, 1):
        foo, bcolors, lcolors = color_branches(surf_mesh)
        assert not bcolors
        # Only keep the longest loop
        if len(lcolors) > 1:
            keep_tag = max(lcolors, key=lambda t, m=foo: loop_len(t, m))
            the_loop = submesh(foo, keep_tag)
        else:
            the_loop = surf_mesh

        assert is_loop(the_loop)
        assert has_unique_vertices(the_loop)

        loops.append(the_loop)

    # FIXME: here we could possibly fix the loop meshes e.g. by kicking
    # out the smallest cells

    # # FIXME: how the surfaces are layed out should be done properly and
    # # structured of how to use nest them to define the volumes should be
    # # done based on graph theory. But what we do here is simply assume
    # # that the largest surface encloses the smallest ones
    loops = sorted(loops, key=lambda mesh: loop_len(None, mesh), reverse=True)

    gmsh.initialize(['', '-clscale', str(scale)])
    model = gmsh.model
    factory = model.occ

    # Here we aim to remesh the surface based on bounding curves. There are
    # several ways to get the curves defined
    curve_loops = []
    for tag, loop in enumerate(loops, 1):
        x = loop.coordinates()
        # There are severral ways to define the curves
        # 1) I prefer this one where we reparatrize the surface by fitting
        # spline to it
        mapping = np.array([factory.addPoint(*xi, z=0) for xi in x])
        loop_vtx = mapping[list(walk_vertices(loop))]
        lines = [factory.addBSpline(loop_vtx)]

        # 2) We reuse vertices of the original mesh - this essentially determines
        # the mesh size and in addition the curves may be wrong
        # cells = mapping[loop.cells()[list(walk_cells(loop))].flatten()].reshape((-1, 2))
        # lines = [factory.addLine(*c) for c in cells]

        factory.synchronize()
        model.addPhysicalGroup(1, lines, tag)

        curve_loops.append(factory.addCurveLoop(lines))

    surf = factory.addPlaneSurface(curve_loops)
    factory.synchronize()
    model.addPhysicalGroup(2, [surf], 1)

    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    nodes, topologies = msh_gmsh_model(model, 2)

    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions


def hull_distance(mesh1d, nrefs=0):
    '''Walk with computing distance from convex hull (refined for distance accuracy)'''
    assert nrefs >= 0
    
    X = mesh1d.coordinates()
    
    cvx_hull = convex_hull(X)
    # Close the surface
    cvx_hull = np.row_stack([cvx_hull, cvx_hull[0]])

    while nrefs:
        u = np.diff(cvx_hull, axis=0)
        new_pts = cvx_hull[:-1] + 0.5*u
        cvx_hull = np.row_stack(list(zip(cvx_hull, new_pts))+[cvx_hull[-1]])
        nrefs -= 1
        
    return np.array([p for p in walk_vertex_distance_from(mesh1d, cvx_hull)]).reshape((-1, 2)), cvx_hull


def mark_deepest_fold(arcl, depths, tol=1E-10):
    '''(Left, right) indices that are boundaries of the fold'''
    peaks, _ = find_peaks(depths)
    max_peak = peaks[np.argmax(depths[peaks])]
    # Now the fold should be closest neighboring indices who have depth zero
    on_surface = depths < tol

    right = np.where(np.logical_and(on_surface, arcl - arcl[max_peak] > 0))[0][0]
    left = np.where(np.logical_and(on_surface, arcl < arcl[max_peak]))[0][-1]

    assert left < max_peak < right

    return (left, right)


def mark_folds_bylength(arcl, depths, tol=1E-10):
    '''Fold boundaries solded by depth'''
    peaks, = np.where(depths < tol)
    peaks = peaks[np.argsort(arcl[peaks])]

    bounds = list(zip(peaks[:-1], peaks[1:]))
    lengths = [arcl[r]-arcl[l] for l, r in bounds]

    return sorted(zip(bounds, lengths), key=lambda p: p[1], reverse=True)


def mesh_contour(contour, scale=1, fit_spline=True, view=False):
    '''Collection of points encloses a surface'''
    npts, gdim = contour.shape
    assert npts > 3
    assert np.linalg.norm(contour[0] - contour[-1]) < 1E-13
    
    gmsh.initialize(['', '-clscale', str(scale)])
    model = gmsh.model
    factory = model.occ

    mapping = [factory.addPoint(*xi, z=0) for xi in contour[:-1]]
    mapping.append(mapping[0])
    if fit_spline:
        lines = [factory.addBSpline(mapping)]
    else:
        lines = [factory.addLine(*l) for l in zip(mapping[:-1], mapping[1:])]
    
    factory.synchronize()
    model.addPhysicalGroup(1, lines, 1)

    curve_loops = []
    curve_loops.append(factory.addCurveLoop(lines))
        
    surf = factory.addPlaneSurface(curve_loops)
    factory.synchronize()
    model.addPhysicalGroup(2, [surf], 1)
        
    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    nodes, topologies = msh_gmsh_model(model, 2)

    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh


def clip_fold(fold, arcl, depths, left=0.5, right=0.5):
    '''Walk back from the midpoint of the fold to find the clip'''
    ll, uu = fold
    # Midpoint based on halfway arcl distances
    midd = arcl[ll] + 0.5*(arcl[uu] - arcl[ll])
    indices = np.arange(ll, uu+1)
    tip = indices[np.argmin(np.abs(arcl[indices] - midd))]
    
    # How far from tip to right
    right_of = np.arange(tip, uu+1)
    distances = (arcl[right_of] - arcl[tip])/(arcl[uu] - arcl[tip])
    right = right_of[np.argmin(np.abs(distances - right))]

    # How far from tip to left
    left_of = np.arange(ll, tip+1)
    distances = (arcl[tip] - arcl[left_of])/(arcl[tip] - arcl[ll])
    left = left_of[np.argmin(np.abs(distances - left))]

    return left, tip, right


def clip_longest_fold(mesh1d, left=0.75, right=0.75, view=False):
    '''Return a new mesh1d which the longest fold clipped'''
    # A fold is marker by points that are on the surface of the convex
    # hull. We clip by walking relative distace from the deepest place
    # of the fold

    closed_surfaces, tags = connected_domains(mesh1d)
    # Make sure that there is only one bounding surface
    assert len(list(range(*tags))) == 1
    # Only one surface means that we can continue with mesh1d
    
    idx_dist_h, cvx_hull = hull_distance(mesh1d)
    idx, dist_h = idx_dist_h.T
    idx = np.asarray(idx, 'uintp')

    X = mesh1d.coordinates()
    
    _, arcl = np.fromiter(sum(walk_vertex_distance(mesh1d), ()), dtype=float).reshape((-1, 2)).T
    # Get the fold boundaries
    fold, fold_size = mark_folds_bylength(arcl=arcl, depths=dist_h, tol=1E-10)[0]
    print(f'Removing fold with len {fold_size}')
    # Adjust where to clip
    l, tip, r = clip_fold(fold, arcl, dist_h, left=left, right=right)

    # Clip
    contour = np.r_[idx[:l+1], idx[r-1:]]
    contour = X[contour]

    if view:
        plt.figure()
        plt.plot(cvx_hull[:, 0], cvx_hull[:, 1])
        plt.plot(X[idx, 0], X[idx, 1], 'r')
        plt.plot(X[idx[l:r], 0], X[idx[l:r], 1], 'b')

        plt.plot(X[idx[l], 0], X[idx[l], 1], 'ro')
        plt.plot(X[idx[r], 0], X[idx[r], 1], 'go')
        plt.plot(X[idx[tip], 0], X[idx[tip], 1], 'bo')

        plt.axis('equal')
        plt.show()
    
    # Mesh it
    mesh1d = ContourMesh(contour)

    return mesh1d

# ------------------------------------------------------------------------

if __name__ == '__main__':

    mesh_path = './2d_brain_mesh.xml'
    mesh2d = df.Mesh(mesh_path)

    # In './2d_brain_mesh.xml' this is redundant
    mesh2d = remove_null_cells(mesh2d, null=1E-14)
    mesh1d = boundary_mesh(mesh2d)
    df.File('original.pvd') << mesh1d

    # NOTE: left, right represent (x_clip - x_tip)/fold_half_length.
    # That is, the closer we are to 1 the closer the clipping will happen
    # to the surface.
    
    # Lower one
    mesh1d_0 = clip_longest_fold(mesh1d, left=0.85, right=0.905)
    df.File('clipped_0.pvd') << mesh1d_0

    # Right one
    mesh1d_1 = clip_longest_fold(mesh1d_0, left=0.6, right=0.875)
    df.File('clipped_1.pvd') << mesh1d_1
    
    # Front one
    mesh1d_2 = clip_longest_fold(mesh1d_1, left=0.9, right=0.8)
    df.File('clipped_2.pvd') << mesh1d_2

    df.File('clipped_2_mesh.xml') << mesh1d_2
    
    # 0.125 is a decent resolution
    # NOTE: each of the mesh1d_* could be meshed
    mesh_bounded_surface(mesh1d_2, scale=0.125, view=False)
