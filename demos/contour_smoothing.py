from pyvista_contour import as_linked_points, only
from itertools import groupby
import pyvista    

from itertools import chain
import numpy as np
from dolfin import *


def solve_heat(alpha, bdries, f, u0, dirichlet_data, neumann_data, dt=0.1, Tfinal=1):
    '''
    Solve du/dt -alpha*Delta u = f starting from u0
    and having 

      u = gD on Dirichlet bdry
      grad(u).n = gN on Neumann bdry
    '''
    assert not dirichlet_data.keys() & neumann_data.keys()

    mesh = bdries.mesh()
    V = FunctionSpace(mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)

    u0 = interpolate(u0, V)
    alpha, dt = Constant(alpha), Constant(dt)

    ds = Measure('ds', domain=mesh, subdomain_data=bdries)
    dx = Measure('dx', domain=mesh)

    a = inner(u, v)*dx + alpha*dt*inner(grad(u), grad(v))*dx
    L = inner(u0, v)*dx + dt*inner(f, v)*dx

    # Neumann contrib
    for tag, value in neumann_data.items():
        L += dt*inner(value, v)*ds(tag)

    # Dirichlet bcs
    bcs = [DirichletBC(V, value, bdries, tag) for tag, value in dirichlet_data.items()]
    # In this problem the system matrix stays the same and only rhs is
    # modified. 
    assembler = SystemAssembler(a, L, bcs)
    A, b = Matrix(), Vector()
    # Therefore we want to build the matrix here and set up
    # the solver for it only once here
    assembler.assemble(A)

    # solver = LUSolver(A)
    # For large meshes we might need to go iterative
    print(f'Solving for {V.dim()} unknowns')
    solver = PETScKrylovSolver('cg', 'hypre_amg')
    solver.set_operators(A, A)
    ksp_params = solver.parameters
    ksp_params['relative_tolerance'] = 1E-10
    ksp_params['absolute_tolerance'] = 1E-8
    ksp_params['nonzero_initial_guess'] = True
    ksp_params['monitor_convergence'] = True    # this one is verbose

    t = 0
    while t < Tfinal:
        # Updata bdry data
        for key, val in chain(neumann_data.items(), dirichlet_data.items()):
            val.t = t
        # AND source
        f.t = t
        # The new RHS
        assembler.assemble(b)
        # Solve into "previous time step"
        solver.solve(u0.vector(), b)

        t += dt(0)

        yield t, u0


def get_contourlines(f, values):
    '''Get contour lines of f corresponding to values'''
    # NOTE: even if there is just one value this can consist of
    # several curves. We return a groupby iterator cval -> pieces
    
    mesh = f.function_space().mesh()
    
    # Use pyvista to find the contour
    cells = mesh.cells()
    num_cells, vtx_per_cell = cells.shape
    offsets = vtx_per_cell*np.arange(num_cells)
    # vtk.VTK_TRIANGLE is 5
    cell_types = 5*np.ones(num_cells, dtype='uintp')

    x, y = mesh.coordinates().T
    coordinates = np.c_[x, y, np.zeros_like(x)]

    grid = pyvista.UnstructuredGrid({5: cells}, coordinates)
    # Represent function on mesh
    grid.point_arrays['f'] = f.compute_vertex_values()
    # grid.plot(show_edges=True)
    
    # Get contour corresponding to the level set
    # NOTE: for neuron we would be working with just one level set value
    # but this is here to check generality of the linking idea
    contours = grid.contour(values)
    # 0.25 gives us a tricky case
    #  /\  /\
    # /  \/  \    We diagonose it as two loops. These is stricly speaking
    # \  /\  /    one contour. If we need to group them one option is to 
    #  \/  \/     condsider point values so
    # contours.plot(show_edges=True)

    contour_values = contours.point_arrays['f']
    
    X, pieces = as_linked_points(contours)
    # We are in plane
    X = X[:, :-1]
    # We want to color the piece of countour by same color if it was
    # part of one level set
    pieces = groupby(pieces, key=lambda x: only(set(np.round(contour_values[x], 8))))

    return X, pieces


def contour_len(X, points):
    '''Length of contour defined by X coordinates and list of points'''
    return np.linalg.norm(np.linalg.norm(np.diff(X[points], axis=0), 2, axis=1))

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    from bastian_inside_folds import mesh_bounded_surface, walk_vertex_distance
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

    forced = False
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
    alpha = 0.2
    f = Constant(0)

    for t, u0 in solve_heat(alpha, bdries, f, u0, dirichlet_data, neumann_data):
        # NOTE: maybe you want to store at intermediate times 
        continue

    # When we are done with smoothing we perhaps want to try with several
    # "temperatures" so we store the field so that smoothing needs not to
    # be repeated and we can resume the pipeline from here on

    with XDMFFile(heat_mesh.mpi_comm(), 'smoothed.xdmf') as file:
        file.write_checkpoint(u0, "u0", 0, XDMFFile.Encoding.HDF5, append=False)

    with XDMFFile(heat_mesh.mpi_comm(), 'smoothed_mesh.xdmf') as file:
        file.write(heat_mesh)
    
    # Here's where we'd resume        
    heat_mesh = Mesh(MPI.comm_self)
    with XDMFFile(heat_mesh.mpi_comm(), 'smoothed_mesh.xdmf') as file:
        file.read(heat_mesh)
        
    V = FunctionSpace(heat_mesh, 'CG', 1)
    u0 = Function(V)
    # ... also load the data
        
    with XDMFFile(heat_mesh.mpi_comm(), 'smoothed.xdmf') as file:
        file.read_checkpoint(u0, "u0", 0)    

    # Check the smoothing effect
    plt.figure()
    # Original
    plt.plot(inside_contour[:, 0], inside_contour[:, 1])
    # Smoothed
    cvalues = [0.125]

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

    # It reamains to make the mesh as in `bastian_inside_fold`
    cmesh, _ = ContourMesh(X[contour_piece])
    
    smesh, _ = mesh_bounded_surface(cmesh, scale=1/2**4, view=False)

    File('smoothed_domain.xml') << smesh
    # Let's try to preseve the volume
    # The original is the |bbox volume| =  |outside| + |inside|
    vol0 = np.prod(ur-ll) - assemble(Constant(1)*dx(domain=heat_mesh))
    vol_smoothed = assemble(Constant(1)*dx(domain=smesh))
    print(f'Target volume {vol0}')
    smesh.coordinates()[:] *= sqrt(vol0/vol_smoothed)
    print(f'Rescaled smoothed volume volume {assemble(Constant(1)*dx(domain=smesh))}')

    File('smoothed_domain.pvd') << smesh

    # Finally get the normal vector 
    from slash.utils import surface_normal_vector

    xn, nx = surface_normal_vector(smesh)

    with HDF5File(smesh.mpi_comm(), f'axial_mesh.h5', 'w') as out:
        out.write(smesh, 'mesh')
