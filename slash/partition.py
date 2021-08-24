from collections import defaultdict, namedtuple
import dolfin as df
import numpy as np
import pymetis
# https://github.com/inducer/pymetis


def mesh2graph(mesh):
    '''
    Graph representation of the mesh. Cell is a node, edge is determined 
    by facet connectivity
    '''
    tdim = mesh.topology().dim()
    fdim = tdim - 1

    _, c2f = mesh.init(tdim, fdim), mesh.topology()(tdim, fdim)    
    _, f2c = mesh.init(fdim, tdim), mesh.topology()(fdim, tdim)
    # Cell to cell connectivity in terms of facets 
    adj_list = [np.fromiter(set(sum((f2c(f).tolist() for f in c2f(c)), [])) - set((c, )), dtype='uintp')
                for c in range(mesh.num_cells())]
    
    return adj_list


def partition(mesh, nsubs):
    '''
    Color mesh by mesh partitioner in `nsubs` subdomains. Return cell function
    colored by subdomains, facet function marking interfaces and lookup 
    table for who the interface belongs to.
    '''
    g = mesh2graph(mesh)
    opts = pymetis.Options()
    opts.contig = True
    ncuts, coloring = pymetis.part_graph(nsubs, adjacency=g, options=opts)

    coloring = np.array(coloring)
    ncolors = len(np.unique(coloring))

    cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 1)
    values = cell_f.array()
    for color in range(ncolors):
        values[coloring == color] = color + 1

    interfaces = _find_interfaces(cell_f, ncolors)
    
    return cell_f, interfaces


def _find_interfaces(cell_f, nsubs):
    '''Mark interfaces. Interface is internal between subdomains or exterior'''
    tdim = cell_f.dim()
    mesh = cell_f.mesh()
    assert mesh.topology().dim() == tdim

    _, c2f = mesh.init(), mesh.topology()(tdim, tdim-1)
    _, f2c = mesh.init(), mesh.topology()(tdim-1, tdim)

    cell_f = cell_f.array()

    interfaces = defaultdict(list)
    for color in range(1, 1+nsubs):
        cells_of_color, = np.where(cell_f == color)
        all_facets = np.unique(np.hstack([c2f(c) for c in cells_of_color]))
        for f in all_facets:
            cells_of_facet = f2c(f)
            # Physical boundary
            if len(cells_of_facet) == 1:
                interfaces[(color, )].append(f)
            else:
                color0, color1 = cell_f[cells_of_facet]
                # Internal interface
                if color0 != color1:
                    interfaces[tuple(sorted((color0, color1)))].append(f)
                    
    # Translate to facet_function
    facet_f = df.MeshFunction('size_t', mesh, tdim-1, 0)
    values = facet_f.array()

    lookup = {}
    
    next_color = 1
    for key in sorted(interfaces):
        # Exterior get color of the subdomain
        if len(key) == 1:
            color = key[0]
        # Interior is first come first serve
        else:
            color = next_color
        
        values[interfaces[key]] = color
        lookup[color] = key
        
        next_color += 1

    return namedtuple('Interface', ('facet_f', 'lookup'))(facet_f, lookup)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    
    mesh = df.UnitSquareMesh(32, 32)

    # from gmshnics import gUnitSquare
    # mesh, _ = gUnitSquare(0.2)

    color_f, interface = partition(mesh, 3)

    df.File('foo.pvd') << color_f
    df.File('bar.pvd') << interface.facet_f
