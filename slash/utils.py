from slash.embedded_mesh import EmbeddedMesh
from slash.convex_hull import convex_hull, Point

import networkx as nx
import dolfin as df
import numpy as np


def remove_null_cells(mesh, null=1E-14):
    '''Create new mesh with small(null) cells removed'''
    markers = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

    for cell in df.cells(mesh):
        markers[cell] = int(cell.volume() > null)

    new_mesh = submesh(markers, 1)
    print(quality_volume(mesh, new_mesh))
    return new_mesh

    
def submesh(markers, values):
    '''Submesh of marked cells'''
    mesh = markers.mesh()
    assert markers.dim() == mesh.topology().dim()

    markers_arr = markers.array()
    if isinstance(values, int):
        values = (values, )
    values = iter(values)

    tag = next(values)
    for other_tag in values:
        markers_arr[markers_arr == other_tag] = tag

    new_mesh = df.SubMesh(mesh, markers, tag)
    print('Keeping {}/{} cells'.format(new_mesh.num_cells(), mesh.num_cells()))

    return new_mesh

    
def quality_volume(*meshes):
    '''Did things improve?'''
    rr = df.MeshQuality.radius_ratio_min_max
    vv = lambda mesh: (lambda vec: (vec.min(), vec.max()))(df.as_backend_type(volume_function(mesh).vector()))
    
    stats = (tuple(rr(mesh)) + vv(mesh) + (mesh.num_cells(), ) for mesh in meshes)

    return tuple(stats)


def volume_function(arg, tag=None):
    '''P0 function that has mesh sizes'''
    if isinstance(arg, df.Mesh):
        cell_f = df.MeshFunction('size_t', arg, arg.topology().dim(), 0)
        return volume_function(cell_f, tag=(0, ))

    if isinstance(tag, int):
        return volume_function(arg, (tag, ))

    mesh = arg.mesh()
    Q = df.FunctionSpace(mesh, 'DG', 0)
    q = df.TestFunction(Q)
    v = df.Function(Q)
    dx_ = df.Measure('dx', domain=mesh, subdomain_data=arg)
    df.assemble(sum(q*dx_(t) for t in tag), tensor=v.vector())

    return v


def entity_mesh(entity_f, tags):
    '''Embedded mesh of entities where entitt_f == tags'''
    return EmbeddedMesh(entity_f, tags)


def boundary_mesh(mesh):
    '''Topological boundary'''
    facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    df.DomainBoundary().mark(facet_f, 1)

    return entity_mesh(facet_f, 1)


def connected_domains(mesh):
    '''A cell function colored by connected components of the mesh'''
    tdim = mesh.topology().dim()
    _, f2c = mesh.init(tdim-1, tdim), mesh.topology()(tdim-1, tdim)
    c2f = mesh.topology()(tdim, tdim-1)
    
    graph = nx.Graph()
    graph.add_edges_from(tuple(c2f(c)) for c in range(mesh.num_cells()))

    cell_f = df.MeshFunction('size_t', mesh, tdim, 0)
    values = cell_f.array()
    for tag, cc in enumerate(sorted(nx.algorithms.connected_components(graph)), 1):
        cells_cc = np.unique(np.hstack([f2c(f) for f in cc]))
        values[cells_cc] = tag

    return cell_f, (1, tag+1)


def approx_enclosed_volume(mesh):
    '''Of a loop by convex hull'''
    assert mesh.geometry().dim() == 2
    assert mesh.topology().dim() == 1

    x = mesh.coordinates()
    hull_pts = convex_hull([Point(xi, yi) for xi, yi in zip(*x.T)])
    
    hull = np.c_[[p.x for p in hull_pts], [p.y for p in hull_pts]]
    center = np.mean(hull, axis=0)

    hull = np.row_stack([hull, hull[0]])

    tri_area = lambda A, B, C: 0.5*np.abs(np.cross(B-A, C-A))

    return sum(tri_area(p, q, center) for p, q in zip(hull[:-1], hull[1:]))    


def has_unique_vertices(mesh):
    '''Does it?'''
    assert mesh.geometry().dim() == 2
    x, y = mesh.coordinates().T
    # This is a neat trick :)
    xy = x + 1j*y
    return len(np.unique(xy)) == len(xy)


def fit_ellipse(xy):
    '''Center, major and minor'''
    x, y = xy.T
    x0, y0 = x.mean(), y.mean()
    x = x - x0
    y = y - y0

    (A00, A01, A11) = np.linalg.lstsq(np.c_[x**2, 2*x*y, y**2], np.ones_like(x))[0]
    A = np.array([[A00, A01], [A01, A11]])
    vals, vecs = np.linalg.eigh(A)

    vals = abs(vals)

    a_max = 1./np.sqrt(vals[0])
    vec_max = vecs[:, 0]

    a_min = 1./np.sqrt(vals[1])
    vec_min = vecs[:, 1]

    if a_max < a_min:
        a_min, a_max = a_max, a_min
        vec_min, vec_max = vec_max, vec_min

    center = np.array([x0, y0])

    return center, (a_max, vec_max), (a_min, vec_min)
