from slash.embedded_mesh import EmbeddedMesh
from slash.partition import mesh2graph
from itertools import count, dropwhile
from collections import defaultdict
import networkx as nx
import dolfin as df
import numpy as np


def FacetFunction(mesh, init_c=0):
    '''Size_t of tdim-1'''
    return df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, init_c)


def CellFunction(mesh, init_c=0):
    '''Size_t of tdim'''
    return df.MeshFunction('size_t', mesh, mesh.topology().dim(), init_c)


def cell_graph(cell_f, color=None):
    '''Wrap for networkx'''
    # For mesh work with monochromatic
    if isinstance(cell_f, df.Mesh):
        cell_f = CellFunction(cell_f)
        return cell_graph(cell_f, color=0)
    
    mesh = cell_f.mesh()
    
    tdim = mesh.topology().dim()
    fdim = tdim - 1

    _, c2f = mesh.init(tdim, fdim), mesh.topology()(tdim, fdim)    
    _, f2c = mesh.init(fdim, tdim), mesh.topology()(fdim, tdim)

    g = nx.Graph()
    # Nodes are cells
    g.add_edges_from(filter(lambda cs: len(cs) == 2 and all(cell_f[c] == color for c in cs),
                            (f2c(f)
                             for c in df.SubsetIterator(cell_f, color)
                             for f in c.entities(fdim))))
    
    return g


def as_parent_entities(entities, child_mesh, parent_mesh):
    '''Return mapped entities'''
    chdim = child_mesh.topology().dim()
    pdim = parent_mesh.topology().dim()
    
    mapping = child_mesh.parent_entity_map
    assert parent_mesh.id() in mapping
    mapping = mapping[parent_mesh.id()]

    # In this application we will have two cases.
    # One is where we want to represent cells of child mesh as facets
    # in the other.
    if chdim == pdim - 1:
        child_cell2parent_facet = mapping[chdim]
        return [child_cell2parent_facet[e] for e in entities]

    assert chdim == pdim

    child_cell2parent_cell = mapping[pdim]
    _, f2c_child = child_mesh.init(chdim-1, pdim), child_mesh.topology()(chdim-1, pdim)
    _, c2f_parent = parent_mesh.init(pdim, chdim-1), parent_mesh.topology()(pdim, chdim-1)    
    # The other case is we have mesh < mesh and we want to embed facets
    # We ask the child for facets of a cell, the cell we can map and then
    # wa try to lookup the facet
    mapped = []
    for child_facet in entities:
        child_cell, = f2c_child(child_facet)
        parent_cell = child_cell2parent_cell[child_cell]
        parent_facets = c2f_parent(parent_cell)

        if len(parent_facets) == 1:
            idx, = parent_facets
        else:
            child_facet = df.Facet(child_mesh, child_facet).midpoint()
            parent_facets = [df.Facet(parent_mesh, pf) for pf in parent_facets]
            parent_facet = min(parent_facets, key=lambda l: l.midpoint().distance(child_facet))
            
            assert parent_facet.midpoint().distance(child_facet) < 1E-13, parent_facet.midpoint().distance(child_facet)

            idx = parent_facet.index()
        mapped.append(idx)
    return mapped


def connected_components(f, color):
    '''
    List where weach item is a (cell-indices-of-connected-component 
    of given color, ([facet indices of the boundary]))
    '''
    tdim = f.dim()
    cdim = f.mesh().topology().dim()

    # Get the boundary in pieces representing connected compoenents
    if cdim == tdim + 1:
        surface_mesh = EmbeddedMesh(f, color)

        graph = cell_graph(surface_mesh)
        # Get connected components; it will be wrt to numbering of the
        # surface mesh but we want the map with respect to mesh of f
        return tuple(as_parent_entities(cc, surface_mesh, f.mesh())
                     # Cell of surface mesh as facets of f.mesh()
                     # surface mesh is a (facet) submesh of mesh
                     for cc in nx.algorithms.connected_components(graph))

    assert cdim == tdim

    graph = cell_graph(f, color=color)

    f = df.MeshFunction('size_t', f.mesh(), f.dim(), 0)
    f_values = f.array()
    # We first want to isolate the connected components based on the cell function
    components = []
    for cc in map(list, nx.algorithms.connected_components(graph)):
        # # Mark the cc
        f_values[cc] = 1
        cc_mesh = EmbeddedMesh(f, 1)
        # Now we want to setup the problem for finding the boundary
        cc_facet_f = FacetFunction(cc_mesh)
        df.DomainBoundary().mark(cc_facet_f, 1)
        # We get back the entities in the numbering of submesh
        bdry_ccs = connected_components(cc_facet_f, color=1)
        # What we want is the numbering of the facets as entities of mesh
        # Facets of cc mesh as facets of mesh, cc_mesh is a (cell) submesh of mesh
        bdry_ccs = tuple(as_parent_entities(bdry_cc, cc_mesh, f.mesh()) for bdry_cc in bdry_ccs)
        # And the you have it (indices of cell, (indices of boundaries))
        components.append((cc, (bdry_ccs)))
        # Reset for next on
        f_values[cc] = 0

    return components


def mark_connected_components(cell_f, tag=None):
    '''
    Cell function marking connected components, facet marking cc boundaries 
    and a lookup table {color -> {colored cc originally of color -> colored boundaries}
    '''
    tags = set(cell_f.array())
    assert len(tags) > 1    
    if tag is not None:
        tags = [tag]

    mesh = cell_f.mesh()
    cc, cc_boundaries, lookup = CellFunction(mesh), FacetFunction(mesh), {}
    # Want to have a mapping of which islands descend from one color
    tag_children = defaultdict(list)
    
    cc_array, cc_boundaries_array = cc.array(), cc_boundaries.array()
    cc_colors, cc_bdry_colors = count(1), count(1)

    visited = {}
    is_visited = lambda facets, d=visited: (next(
        dropwhile(lambda tag: not any(f in d[tag] for f in facets), d)
    ), )

    for itag, tag in enumerate(tags):
        for cc_idx, cc_bdry_indices in connected_components(cell_f, tag):
            cc_color = next(cc_colors)
            cc_array[cc_idx] = cc_color

            cc_bdry_colors_ = set()
            for cc_bdry_index in cc_bdry_indices:
                # For the first tag, since the connected components are
                # distince the boundaries do not need to check for the
                # duplicates
                if itag == 0:
                    bdry_color = 0
                else:
                    bdry_color = is_visited(cc_bdry_index)
                    
                if not bdry_color:
                    bdry_color = next(cc_bdry_colors)
                    visited[bdry_color] = set(cc_bdry_index)
                else:
                    bdry_color, = bdry_color
                    
                cc_bdry_colors_.add(bdry_color)
                cc_boundaries_array[cc_bdry_index] = bdry_color

            lookup[cc_color] = cc_bdry_colors_
            tag_children[tag].append(cc_color)

    return cc, cc_boundaries, lookup, tag_children


def close_1surf_bounded_volumes(cell_f, facet_f, lookup, predicate=lambda v, vc: True):
    '''Merge with the surrounding'''
    found = False
    
    cell_f_arr, facet_f_arr = cell_f.array(), facet_f.array()
    items = list(lookup.items())
        
    for vol_color, surf_colors in items:
        # NOTE: the assumption that small volumes have one bounded surface
        if len(surf_colors) == 1:
            vol = EmbeddedMesh(cell_f, vol_color)
            if predicate(vol, vol_color):
                found = True

                surf_color, = surf_colors
                # Change the color based on the surrounding.
                # Neighbor = not us who has the boundary of the same color
                outer = next(dropwhile(lambda tag: not(tag != vol_color and surf_color in lookup[tag]),
                                       lookup))
                cell_f_arr[np.where(cell_f_arr == vol_color)] = outer
                facet_f_arr[np.where(facet_f_arr == surf_color)] = 0

                del lookup[vol_color]
                lookup[outer].remove(surf_color)
                 
    if found:
        return close_1surf_bounded_volumes(cell_f, facet_f, lookup, predicate)
    return cell_f, facet_f, lookup


def close_small_volumes(cell_f, facet_f, lookup, tol):
    '''Remove based on volume'''
    def predicate(vol, vc, tol=tol):
        return df.assemble(df.Constant(1)*df.dx(domain=vol)) < tol

    return close_1surf_bounded_volumes(cell_f, facet_f, lookup, predicate)    

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    
    mesh = df.UnitSquareMesh(8, 8)

    cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 1)

    df.CompiledSubDomain(' & '.join(['(0.125 - tol < x[0])',
                                     '(x[0] < 0.875 + tol)',
                                     '(0.125 - tol < x[1])',
                                     '(x[1] < 0.875 + tol)']), tol=1E-8).mark(cell_f, 2)

    df.CompiledSubDomain(' & '.join(['(0.25 - tol < x[0])',
                                     '(x[0] < 0.75 + tol)',
                                     '(0.25 - tol < x[1])',
                                     '(x[1] < 0.75 + tol)']), tol=1E-08).mark(cell_f, 1)


    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), "/home/mirok/Downloads/021_coronal_16/brain_mesh.h5", "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    hdf.read(subdomains, "/subdomains")
    
    cc, cc_bdries, lookup, _ = mark_connected_components(subdomains)

    df.File('cc.pvd') << cc
    df.File('cc_bdries.pvd') << cc_bdries

    cc, cc_bdries, lookup = close_small_volumes(tol=10, cell_f=cc, facet_f=cc_bdries, lookup=lookup)    
    cc, cc_bdries, lookup = close_small_volumes(tol=100, cell_f=cc, facet_f=cc_bdries, lookup=lookup)

    df.File('closed_cc.pvd') << cc
    df.File('closed_cc_bdries.pvd') << cc_bdries
    
