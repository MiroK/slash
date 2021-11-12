from slash.embedded_mesh import EmbeddedMesh
from mark_within import is_embedded_surface, as_graph
from itertools import combinations
import networkx as nx
import dolfin as df
import numpy as np


def as_graph(cell_f, which):
    '''Graph representation of cells where cell_f == which'''
    mesh = cell_f.mesh()
    
    assert is_embedded_surface(mesh)

    gdim, tdim = mesh.geometry().dim(), mesh.topology().dim()
    mesh.init(tdim-1, tdim)
    f2c = mesh.topology()(tdim-1, tdim)

    cell_f = cell_f.array()
    
    g = nx.Graph()
    for f in range(mesh.num_entities(tdim-1)):
        # If both colors of cells connected to the facet are diffent it's good
        edge = f2c(f)
        colors = cell_f[edge]

        edge = edge[colors == which]

        if len(edge) == 1:
            g.add_node(edge[0])
        else:
            g.add_edges_from(combinations(edge, 2))

    return g


def is_continuous_cover(cell_f, initial):
    '''Cells where cell_f[initial] form this'''
    icolor, = set(cell_f.array()[initial])

    g = as_graph(cell_f, icolor)
    # Single cell
    if g.number_of_nodes() == 1:
        return True
    # Several that are disconneded
    if g.number_of_edges() == 0:
        return False
    # Where we need to decide based on connected components
    return nx.algorithms.number_connected_components(g) == 1
    

def grow_cover(facet_f, itag, checkpoints=None):
    '''Yield facets that grow the patch which is initially facet_f == itag'''
    mesh = facet_f.mesh()
    assert facet_f.dim() == mesh.topology().dim() - 1

    initial, = np.where(facet_f.array() == itag)  # Facets

    bdry_facet = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    df.DomainBoundary().mark(bdry_facet, itag+1)
    bdry_facet.array()[initial] = itag

    emesh = EmbeddedMesh(bdry_facet, (itag, itag+1))
    cell_f = emesh.marking_function
    # Mapping from surface mesh cells to facets of mesh
    c2f = emesh.parent_entity_map[mesh.id()][cell_f.dim()]
    f2c = dict(zip(c2f.values(), c2f.keys()))

    initial = [f2c[f] for f in initial]  # Cells
    for (cover_c, coverage) in huygens_embedded(initial, cell_f, checkpoints=checkpoints, itag=itag):
        cover_f = set(c2f[c] for c in cover_c)
        yield (cover_f, coverage)

        
def huygens_embedded(initial, cell_f, checkpoints=None, itag=1, strict=True):
    '''Grow of cover based on cell function of a surface mesh'''
    mesh = cell_f.mesh()
    assert is_embedded_surface(mesh)
    # Start from what is marked as itag
    assert set(cell_f.array()[initial]) == set((itag, ))
    # It is only this that is marked as itag
    assert set(initial) == set(np.where(cell_f.array() == itag)[0])
    
    # We will return the updated cover each time seeds yield something
    # but that might be too much so here is an option to yield once the
    # relative area of the cover exceeds given thresholds
    if checkpoints is not None:
        assert all(0 < cp < 1.1 for cp in checkpoints)
        checkpoints = list(reversed(checkpoints))
    # Will need area for normalization
    total_area = df.assemble(df.Constant(1)*df.dx(domain=mesh))
    
    cdim = cell_f.dim()
    _, c2f = mesh.init(cdim, cdim-1), mesh.topology()(cdim, cdim-1)
    _, f2c = mesh.init(cdim-1, cdim), mesh.topology()(cdim-1, cdim)

    c2c = lambda c: np.hstack([f2c(f) for f in c2f(c)])

    cover = set(initial)
    
    cell_f = cell_f.array()

    visited = df.MeshFunction('size_t', mesh, cdim, 0)
    dV = df.Measure('dx', domain=mesh, subdomain_data=visited)
    visited_arr = visited.array()
    visited_arr[cell_f == itag] = 1

    # Require that the initial cover is not patchy
    assert not strict or is_continuous_cover(visited, initial)

    area = df.assemble(df.Constant(1)*dV(1))
    ratio = area/total_area
    # The initial cover ...
    yield (cover, ratio)
    # ... might already be larger than some checkpoints so we kick them out
    if checkpoints:
        while ratio > checkpoints[-1]:
            checkpoints.pop()

    # Front will be the outer boundary of the cover; that is each cell where
    # at least for one edge the connected cell is not an itag
    front = set([c for c in cover if np.any(cell_f[c2c(c)] != 1)])

    while front:
        front = set(np.hstack([c2c(c) for c in front])) - cover
        cover.update(front)

        visited_arr[list(cover)] = 1
        area = df.assemble(df.Constant(1)*dV(1))
        ratio = area/total_area

        if checkpoints is None:
            yield cover, ratio
        else:
            if checkpoints:
                # First time exceed
                if ratio > checkpoints[-1]:
                    yield (cover, ratio)
                    # Kick out
                    checkpoints.pop()

    yield (cover, ratio)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    
    mesh = UnitCubeMesh(8, 8, 8)

    # mesh = Mesh()
    # xdmf = XDMFFile(mesh.mpi_comm(), '/home/mirok/Documents/MkSoftware/haznics_ra/brain_meshes/16_enlarged/mesh.xdmf')
    # xdmf.read(mesh)

    # This would be the function to be used for boundary conditions in
    # the simulation. And supposed that some initial part of is marked
    # such that marked (surface) facets form a patch.
    facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(facet_f, 1)

    marked, = np.where(facet_f.array() == 1)
    # Suppose these are the marked facet indices ...
    initial = [marked[0]]
    facet_f.set_all(0)
    # and we mark the patch
    facet_f.array()[initial] = 1

    array = facet_f.array()
    # Now we want to grow the patch in a way that we get 10, 25, 50, 75, 100% coverage
    checkpoints = [0.1, 0.25, 0.5, 0.75, 1.0]
    covers = grow_cover(facet_f=facet_f, itag=1, checkpoints=checkpoints)

    out = df.File('cover.pvd')
    for time, (cover, coverage) in enumerate(covers):
        print(len(cover), coverage)
        # We mark with one the ever growing domain of boundary conditions
        array[list(cover)] = 1
        out << facet_f, time  # Make it look transient
