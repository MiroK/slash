# -------------------------|
# |                        |
# |    +++++               | 
# |---+-----+---------|    | 
#     +      +        |    | 
#      +    +         |    |
# |----+----+---------|    | 
# |     ++++               |
# |------------------------|
#
# We are given a facet function of the mesh and a sphere. The goal is
# then mark facets within the ball (maybe also limiting to facets of some
# color). We then expect that this inside/outside coloring breaks the
# surface to disjoint surfaces and we want to connect with the colored
# ones that neighbor it but are small
from slash.embedded_mesh import EmbeddedMesh
from itertools import combinations
import networkx as nx
import dolfin as df
import numpy as np


def is_embedded_surface(mesh):
    '''Is mesh an embedded surface'''
    gdim, tdim = mesh.geometry().dim(), mesh.topology().dim()
    return gdim == tdim + 1


def is_closed_surface(mesh):
    '''Is embedde mesh a closed surface'''
    if not is_embedded_surface(mesh):
        return False

    tdim = mesh.topology().dim()
    mesh.init(tdim-1, tdim)
    f2c = mesh.topology()(tdim-1, tdim)
    return all(len(f2c(f)) > 1 for f in range(mesh.num_entities(tdim-1)))


def as_graph(cell_f, avoid):
    '''Represent surface mesh as graph avoiding some cells of it'''
    mesh = cell_f.mesh()
    
    assert is_embedded_surface(mesh)

    gdim, tdim = mesh.geometry().dim(), mesh.topology().dim()
    mesh.init(tdim-1, tdim)
    f2c = mesh.topology()(tdim-1, tdim)

    cell_f = cell_f.array()
    
    g, bdry = nx.Graph(), []
    for f in range(mesh.num_entities(tdim-1)):
        # If both colors of cells connected to the facet are diffent it's good
        edge = f2c(f)
        colors = cell_f[edge]
        if avoid not in list(colors):
            g.add_edges_from(combinations(edge, 2))
        else:
            # We want the okay cell that neigbors avoid
            bdry.extend(edge[colors != avoid])

    return g, bdry


def mark_within(facet_f, centers, radii,  mark_color, restrict_color=None):
    '''
    Return groups of facets (ids) that are with the spheres(centers, radiii) 
    with neighboring connected components 
    '''
    # Use entire boundary if not specified
    if restrict_color is None:
        restrict_color = mark_color+1
        df.DomainBoundary().mark(facet_f, restrict_color)

    # We do everywhing only with facets so it's more efficient to treat
    # them as cells of an embeded mesh
    surface_mesh = EmbeddedMesh(facet_f, restrict_color)
    assert is_closed_surface(surface_mesh), df.File('surface_mesh.pvd') << surface_mesh

    assert len(centers) == len(radii)
    cell_f = surface_mesh.marking_function
    array = cell_f.array()
    inside_balls = []
    
    for center, radius in zip(centers, radii):
        # Reset for new ball
        array[:] = restrict_color
        
        assert len(center) == surface_mesh.geometry().dim()
        if len(center) == 2:
            char_foo = df.CompiledSubDomain('(x[0]-x0)*(x[0]-x0) + (x[1]-x1)*(x[1]-x1) < rad*rad',
                                            rad=radius, x0=center[0], x1=center[1])
        else:
            assert len(center) == 3
            char_foo = df.CompiledSubDomain('(x[0]-x0)*(x[0]-x0) + (x[1]-x1)*(x[1]-x1) + (x[2]-x2)*(x[2]-x2) < rad*rad',
                                            rad=radius, x0=center[0], x1=center[1], x2=center[2])

        char_foo.mark(cell_f, mark_color)
        marked, = np.where(array == mark_color)
        print(f'Marked with ball {(center, radius)} constraint {len(marked)}')
        inside_balls.extend(marked)
    # Union
    inside_balls = np.unique(inside_balls)
    array[inside_balls] = mark_color

    # The idea next is to create a graph that avoids the cell already
    # marked
    g, bdry_cells = as_graph(cell_f, avoid=mark_color)

    # We then want to return cells of subgraphs that form a connected component
    ccs = nx.algorithms.connected_components(g)
    # Makes sense to visit the smallest in terms of area covered ones first
    V = df.FunctionSpace(surface_mesh, 'DG', 0)
    v, area = df.TestFunction(V), df.Function(V)
    df.assemble(df.CellVolume(surface_mesh)*v*df.dx, area.vector())

    area = area.vector().get_local()
    ccs = sorted(ccs, key=lambda cc: sum(area[list(ccs)]))

    # NOTE: ccs are cells of surface mesh and we want them back as facets
    # of the original mesh
    parent = facet_f.mesh()
    c2f = surface_mesh.parent_entity_map[parent.id()][facet_f.dim()]
    c2f = np.array([c2f[c] for c in sorted(c2f)])

    # First we yield those by geometry
    yield c2f[np.where(array == mark_color)[0]]

    bdry_cells = set(bdry_cells)
    # However, that component needs to border our initial patch
    for cc in ccs:
        if bdry_cells & cc:
            # And we want the representation back in the facets of the original mesh
            yield c2f[list(cc)]

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, File, MeshFunction, CompiledSubDomain

    mesh = UnitSquareMesh(128, 128)
    # Want to create similar to the above the initial sketch
    cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    CompiledSubDomain(' && '.join(['(x[0] > 0.25-tol)', '(x[0] < 0.75+tol)',
                                   '(x[1] < 0.75+tol)']), tol=1E-10).mark(cell_f, 1)
    # Make minecraft packman
    mesh = EmbeddedMesh(cell_f, 0)
    # The function where we want to mark the region for bcs
    facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    ff_array = facet_f.array()
    
    mark_color = 100
    #for cc in
    for cc in mark_within(facet_f, centers=[[0.5, 0], [0.5, 0.2]], radii=[0.3, 0.35], mark_color=mark_color):
        # Now we just color the surfaces with different colors to visualize the
        # marking logic. In general I guess we would use the same color and stop
        # at some point
        ff_array[cc] = mark_color
        mark_color += 1

    File('foo.pvd') << facet_f
