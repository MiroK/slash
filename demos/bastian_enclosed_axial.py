from slash.gmsh_interopt import msh_gmsh_model, mesh_from_gmsh
from slash.branching import color_branches, is_loop, walk_vertices
from slash.utils import *

import dolfin as df
import numpy as np

import gmsh, sys


def first(iterable):
    return next(iter(iterable))


def second(iterable):
    iterable = iter(iterable)
    next(iterable)
    return first(iterable)

# FIXME: ellipse fit
# convex hull and curvature flow?

def add_bounding_rectangle(mesh, factory, pad=1.1):
    '''Really a bounding box'''
    assert pad > 1

    x = mesh.coordinates()
    ll = x.min(axis=0)
    ur = x.max(axis=0)

    dx = ur - ll
    ll -= dx*(pad-1)/2.
    ur += dx*(pad-1)/2.

    # Now we draw
    points = (factory.addPoint(ll[0], ll[1], 0),
              factory.addPoint(ur[0], ll[1], 0),
              factory.addPoint(ur[0], ur[1], 0),
              factory.addPoint(ll[0], ur[1], 0))

    n = len(points)
    lines = [factory.addLine(points[i], points[(i+1)%n]) for i in range(n)]
              
    return lines


def add_bounding_ellipse(mesh, factory, pad=0.825):
    '''Enclose in ellipse'''
    X = mesh.coordinates()
    center, (a_max, v_max), (a_min, v_min) = fit_ellipse(X)
    
    # Which guys are outside
    dist = np.sum((X - center)*v_max/a_max, axis=1)**2 + np.sum((X - center)*v_min/a_min, axis=1)**2
    # Scale by looking at the furthest one
    scale = pad*np.max(dist)

    # Defining ellipse
    pts = center + scale*np.array([np.zeros(2),
                                   a_max*v_max,
                                   a_min*v_min,
                                   -a_max*v_max,
                                   -a_min*v_min])

    gpts = np.array([factory.addPoint(x, y, 0) for x, y in pts])
    signs = np.array([1, -1, 1, -1])
    indices = np.array([[2, 0, 1, 1],
                        [4, 0, 1, 1],
                        [4, 0, 3, 3],
                        [2, 0, 3, 3]])

    lines = [factory.addEllipseArc(*gpts[index]) for index in indices]
    # Things for loop are
    return signs*lines

# --------------------------------------------------------------------

if __name__ == '__main__':
    add_bounding_surface = add_bounding_ellipse
    
    mesh_path = '/home/mirok/Documents/MkSoftware/diffusion_simulation_code/simulation_inputs/axial-slice-0.xml'
    mesh2d = df.Mesh(mesh_path)

    print('Starting from', quality_volume(mesh2d))

    # First we remove small elements
    mesh2d = remove_null_cells(mesh2d, null=1E-14)
    # The idea is next to define a geometry based on surfaces that bound
    # the brain 2d surface. So we get the bounding mesh ...
    mesh1d = boundary_mesh(mesh2d)
    # ... and find the bounding surfaces
    closed_surfaces, tags = connected_domains(mesh1d)
    # Here's how it looks
    df.File('volumes.pvd') << closed_surfaces
    
    # Some of these loops practically enclose no volume
    valid_bdries = []
    for tag in range(*tags):
        surf_mesh = entity_mesh(closed_surfaces, tag)
        vol = approx_enclosed_volume(surf_mesh)
        print(vol)
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
    
    # FIXME: how the surfaces are layed out should be done properly and
    # structured of how to use nest them to define the volumes should be
    # done based on graph theory. But what we do here is simply assume
    # that the largest surface encloses the smallest ones
    loops = sorted(loops, key=lambda mesh: loop_len(None, mesh), reverse=True)

    gmsh.initialize(sys.argv)
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

    # We shall also define a bounding surface in terms of the brain surface
    bounding_curves = add_bounding_surface(loops[0], factory)
    bounding_loop = factory.addCurveLoop(bounding_curves)

    factory.synchronize()    
    # Mark it
    tag += 1
    model.addPhysicalGroup(1, np.abs(bounding_curves), tag)    

    
    outside = factory.addPlaneSurface([bounding_loop, curve_loops[0]])
    factory.synchronize()    
    model.addPhysicalGroup(2, [outside], 2)
    
    gmsh.fltk.initialize()
    gmsh.fltk.run()
    
    nodes, topologies = msh_gmsh_model(model,
                                       2,
                                       # NOTE: Globally refine, i.e. this is a way to control the
                                       # coarsenes of the resulting mesh
                                       number_options={'Mesh.CharacteristicLengthFactor': 1./2**5})
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    print('Final mesh', quality_volume(mesh))
    # For paraview    
    df.File('mesh.pvd') << mesh
    df.File('bdries.pvd') << entity_functions[1]
    df.File('subdomains.pvd') << entity_functions[2]
    gmsh.finalize()
    
    # For further use
    df.File('mesh.xml') << mesh
