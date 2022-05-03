import numpy as np
from dolfin import *

from bastian_inside_folds import mesh_bounded_surface, walk_vertex_distance
from contour_meshing import mesh_bounded_contours
from smooth_enclose import solve_heat, get_contourlines, contour_len
from slash.convex_hull import convex_hull
from slash.embedded_mesh import ContourMesh
import matplotlib.pyplot as plt
import os
    
# The idea of this approach for smoothing a 1d in 2d surface is to 
# work with a 2d domain of which the surface is the boundary. On this
# domain we will solve a heat equation where the temperature is fixed
# on the to-be-smoothed surface. Then contours of lower temperatures
# are candidate smoother surfaces.

# Begin be defining the larger domain. Note, that this is expensive
# so ideally do it once and store the mesh
if False:
    pts = np.random.rand(100, 2)
    cvx_hull = convex_hull(pts)
    # Close it
    inside_contour = np.row_stack([cvx_hull, cvx_hull[0]])
else:
    bdry_mesh = Mesh('clipped_2_mesh.xml')
    idx, arcl = np.fromiter(sum(walk_vertex_distance(bdry_mesh), ()), dtype=float).reshape((-1, 2)).T
    idx = idx.astype('uintp')

    inside_contour = bdry_mesh.coordinates()[idx]

# Encloses this shape in a frame to get a geometry on which we
# will do the smoothing
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

forced = True
# Now the contours are used to define a mesh on which we want to solve
# heat equation for smoothing
# Generate
if forced or not all(os.path.exists(f) for f in ('heat_eq_mesh.xml', 'heat_eq_bdries.xml')):
    _, entity_fs = mesh_bounded_surface(cell_f, scale=1/2.**4, view=False, loop_opts={1: False, 2: True})
    bdries = entity_fs[1]
    heat_mesh = bdries.mesh()

    File('heat_eq_mesh.xml') << heat_mesh
    File('heat_eq_bdries.xml') << bdries
# Load
else:
    heat_mesh = Mesh('heat_eq_mesh.xml')
    bdries = MeshFunction('size_t', heat_mesh, 'heat_eq_bdries.xml')

# Now we smooth
dirichlet_data = {2: Constant(1)}
neumann_data = {1: Constant(0)}
u0 = Constant(0)
f = Constant(0)

# Brain
alpha = 0.5

for t, uh in solve_heat(alpha, bdries, f, u0, dirichlet_data, neumann_data):
    # NOTE: maybe you want to store at intermediate times 
    continue

cvalues = [0.95]
X, pieces = get_contourlines(uh, cvalues)
for cval, pieces_of_contour in pieces:
    # But we probably want only the largest one
    clen = lambda pts, X=X: contour_len(X, pts)
    contour_piece = max(pieces_of_contour, key=clen)
# Brain's mesh
brain_mesh, _ = ContourMesh(X[contour_piece])

# Skull
alpha = 10

for t, uh in solve_heat(alpha, bdries, f, u0, dirichlet_data, neumann_data):
    # NOTE: maybe you want to store at intermediate times 
    continue

cvalues = [0.1]
X, pieces = get_contourlines(uh, cvalues)
for cval, pieces_of_contour in pieces:
    # But we probably want only the largest one
    clen = lambda pts, X=X: contour_len(X, pts)
    contour_piece = max(pieces_of_contour, key=clen)
# Brain's mesh
skull_mesh, _ = ContourMesh(X[contour_piece])

surfaces = [(1, 2), (2, )]
contours = {1: skull_mesh, 2: brain_mesh}

for i in range(1, 2):
    smesh, entity_fs = mesh_bounded_contours(contours, surfaces, scale=1/2**i, view=False)
    print(sum(entity_fs[1].array() == 2), '<<')
    with HDF5File(MPI.comm_world, f'brain_skull_{i}.h5', 'w') as out:
        out.write(smesh, 'mesh')
        out.write(entity_fs[2], 'volumes')
        out.write(entity_fs[1], 'surfaces')

File('brain_skull.pvd') << smesh
File('brain_skull_subdomains.pvd') << entity_fs[2]    
File('brain_skull_interfaces.pvd') << entity_fs[1]
