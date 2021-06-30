from slash.gmsh_interopt import msh_gmsh_model, mesh_from_gmsh
from slash.branching import color_branches, is_loop, walk_vertices
from slash.utils import *

import dolfin as df
import numpy as np

import gmsh, sys


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
        
factory.synchronize()
    
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

gmsh.finalize()

# For further use
df.File('mesh.xml') << mesh

# Let's also write things for HDF5
h5 = df.HDF5File(mesh.mpi_comm(), 'mesh.h5', 'w')
h5.write(mesh, 'mesh')
for dim in entity_functions:
    h5.write(entity_functions[dim], f'entity_f_{dim}')
h5.close()

# Setting up bcs with the facet functions read from file
mesh = df.Mesh()
with df.HDF5File(mesh.mpi_comm(), 'mesh.h5', 'r') as h5:
    h5.read(mesh, 'mesh', False)

    tdim = mesh.topology().dim()    
    facet_f = df.MeshFunction('size_t', mesh, tdim-1, 0)
    h5.read(facet_f, f'entity_f_{tdim-1}')

V = df.FunctionSpace(mesh, 'CG', 1)

# Pick marker of one of the tagged surfaces
tag = 1
bc = df.DirichletBC(V, df.Constant(0), facet_f, tag)
# Check that it works
ncstr_dofs = len(bc.get_boundary_values())
assert ncstr_dofs > 0
print(f'Tag {tag} constrains {ncstr_dofs} dofs')
