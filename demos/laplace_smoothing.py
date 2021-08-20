from dolfin import *


def laplace_smooth_mesh(mesh, alpha=0.0, beta=0.5, nsmooths=1):
    assert mesh.topology().dim() == 1
    
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    u, q = TrialFunction(V), TestFunction(Q)

    K = CellVolume(bdry_mesh)
    m = (2/K)*inner(u, q)*dx
    M = PETScMatrix()
    assemble(m, M)

    M_ = M.mat()
    A = M_.transposeMatMult(M_)
    d = A.getDiagonal()
    d.reciprocal()

    A.diagonalScale(d)
    # Do not consider the diagonal point
    d *= 0.
    A.setDiagonal(d)

    A = PETScMatrix(A)

    # Original coordinates
    os = [interpolate(Expression(f'x[{i}]', degree=1), V).vector() for i in range(mesh.geometry().dim())]
    # Coordinates that we will smooth
    qs = [oi.copy() for oi in os]
    # The smoothed coordinates
    ps = [oi.copy() for oi in os]

    k = 0
    while k < nsmooths:
        # Laplace smooth step q-> p
        [A.mult(qi, pi) for qi, pi in zip(qs, ps)]
        # Cobine with original positions
        #bs = [pi - (alpha*oi + (1-alpha)*qi) for oi, qi, pi in zip(os, qs, ps)]
        # Smooth b -> q
        # [A.mult(bi, qi) for bi, qi in zip(bs, qs)]
        # Combine the smoothed ones
        # [pi.axpy(-1.0, beta*bi + (1-beta)*qi) for pi, bi, qi in zip(ps, bs, qs)]

        # Next round
        k += 1
        
    d2v = dof_to_vertex_map(V)

    for i, xi in enumerate(ps):
        mesh.coordinates()[d2v, i] = xi.get_local()

    print(mesh.coordinates())
        
    return mesh

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from xii import EmbeddedMesh
    
    bdry_mesh = Mesh('clipped_2_mesh.xml')
    # The idea is

    m = UnitSquareMesh(4, 4, 'crossed')
    f = MeshFunction('size_t', m, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    CompiledSubDomain('near(1-x[0], x[1])').mark(f, 1)
    CompiledSubDomain('near(0.5, x[1])').mark(f, 1)
    CompiledSubDomain('near(0.5, x[0])').mark(f, 1)

    bdry_mesh = EmbeddedMesh(f, 1)

    V = FunctionSpace(bdry_mesh, 'CG', 1)
    Q = FunctionSpace(bdry_mesh, 'DG', 0)
    u, q = TrialFunction(V), TestFunction(Q)

    K = CellVolume(bdry_mesh)
    m = (2/K)*inner(u, q)*dx
    M = PETScMatrix()
    assemble(m, M)

    M_ = M.mat()
    A = M_.transposeMatMult(M_)
    d = A.getDiagonal()
    d.reciprocal()

    A.diagonalScale(d)

    d *= 0.
    A.setDiagonal(d)
    
    X = (M.array().T).dot(M.array())
    from collections import defaultdict

    v2v = defaultdict(set)
    for cell in cells(bdry_mesh):
        v0, v1 = cell.entities(0)
        v2v[v0].add(v1)
        v2v[v1].add(v0)

    v2d = vertex_to_dof_map(V)
    for i in range(V.dim()):
        dof = v2d[i]

        target = tuple(v2d[v] for v in v2v[i])
        indices, values = A.getRow(dof)
        indices = tuple(indices)

        assert all(t in indices for t in target)
        assert all(abs(values[indices.index(t)] - 1./len(target)) < 1E-13 for t in target), (target, values)
    # Adjecency matrix 

    File('orignal.pvd') << bdry_mesh
    for i in range(2):
        bdry_mesh = laplace_smooth_mesh(bdry_mesh)
        File(f'smooth_s{i}.pvd') << bdry_mesh
