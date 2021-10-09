from slash.gmsh_interopt import msh_gmsh_model, mesh_from_gmsh
from slash.branching import walk_vertices
import numpy as np
import gmsh


def mesh_bounded_contours(contours, surfaces, scale=1., view=False, loop_opts=None):
    '''Mesh the inside bounded by surfaces'''
    gmsh.initialize(['', '-clscale', str(scale)])
    model = gmsh.model
    factory = model.occ

    if loop_opts is None: loop_opts = {}

    # Color loops 
    curve_loops = {}
    for tag, loop in contours.items():
        x = loop.coordinates()
        mapping = np.array([factory.addPoint(*xi, z=0) for xi in x])

        if loop_opts.get(tag, True):
            # There are severral ways to define the curves
            # 1) I prefer this one where we reparatrize the surface by fitting
            # spline to it
            loop_vtx = mapping[list(walk_vertices(loop))]
            lines = [factory.addBSpline(loop_vtx)]
        else:
            # 2) We reuse vertices of the original mesh - this essentially determines
            # the mesh size and in addition the curves may be wrong
            cells = [c[0] for c in walk_cells(loop)]
            cells = mapping[(loop.cells()[cells]).flatten()].reshape((-1, 2))
            lines = [factory.addLine(*c) for c in cells]
            
        factory.synchronize()
        model.addPhysicalGroup(1, lines, tag)

        curve_loops[tag] = factory.addCurveLoop(lines)
    factory.synchronize()

    # Build surfaces in terms of loops
    for tag, pieces in enumerate(surfaces, 1):
        print(tag, pieces)
        surf = factory.addPlaneSurface([curve_loops[p] for p in pieces])
        factory.synchronize()
        model.addPhysicalGroup(2, [surf], tag)
    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    nodes, topologies = msh_gmsh_model(model, 2)

    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions
