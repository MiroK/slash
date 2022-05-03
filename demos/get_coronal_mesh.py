from slash.cc_coloring import close_small_volumes, mark_connected_components
from slash.embedded_mesh import EmbeddedMesh
from slash.branching import color_branches
from contour_meshing import mesh_bounded_contours
from preprocess_coronal import smooth
import dolfin as df
import numpy as np


mesh = df.Mesh()
hdf = df.HDF5File(mesh.mpi_comm(), "./coronal_mesh.h5", "r")   # Coronal

hdf.read(mesh, "/mesh", False)
subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
hdf.read(subdomains, "/subdomains")

cc, cc_bdries, lookup, _ = mark_connected_components(subdomains)

cc, cc_bdries, lookup = close_small_volumes(tol=10, cell_f=cc, facet_f=cc_bdries, lookup=lookup)    

# The one that needs to be smoothed
inner_contour = (2, )
inner_contour_mesh = EmbeddedMesh(cc_bdries, inner_contour)
inner_contour_mesh, _ = smooth(inner_contour_mesh, temperature=0.75, prefix='inner', forced=False)

outer_contour = (1, 3, 4, 5, 6, 7)
outer_contour_mesh = EmbeddedMesh(cc_bdries, outer_contour)
cell_f, branch_colors, loop_colors = color_branches(outer_contour_mesh)

outer_contour_mesh = EmbeddedMesh(cell_f, tuple(set(branch_colors) - set(loop_colors)))

surfaces = [(1, 2), (2, )]
contours = {1: outer_contour_mesh, 2: inner_contour_mesh}

# Different refiments
for i in range(2, 3):
    smesh, entity_fs = mesh_bounded_contours(contours, surfaces, scale=1/2**i, view=False)
    print(smesh.hmin(), sum(entity_fs[1].array() == 2), '<<')

    with df.HDF5File(df.MPI.comm_world, f'synthetic_brain_{i}.h5', 'w') as out:
        out.write(smesh, 'mesh')
        out.write(entity_fs[2], 'volumes')
        out.write(entity_fs[1], 'surfaces')

freq = synthetic_brain.get('frequency', 0)

df.File(f'synthetic_brain_subdomains_{freq}.pvd') << entity_fs[2]    
df.File(f'synthetic_brain_interfaces_{freq}.pvd') << entity_fs[1]
