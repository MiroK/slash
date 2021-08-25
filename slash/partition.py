from slash.embedded_mesh import EmbeddedMesh
from collections import defaultdict, namedtuple
from itertools import takewhile
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


def partition(mesh, nsubs, with_subd_meshes=False):
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

    tdim = mesh.topology().dim()
    cell_f = df.MeshFunction('size_t', mesh, tdim, 1)
    values = cell_f.array()
    for color in range(ncolors):
        values[coloring == color] = color + 1

    interfaces = _find_interfaces(cell_f, ncolors)

    if not with_subd_meshes:
        return cell_f, interfaces

    # Some more work left to do if we want submeshes. We actually want
    # on each subdomain a facet function which has the right color for
    # the interfaces. If the subdomain has only exterior boundary it
    # will take colors from the subdomain color and then we are done
    # so start there
    subdomains, lc2lf_maps, gc2lc_maps = {}, {}, {}
    
    for subd_color in set(sum(interfaces.lookup.values(), ())):
        subdomain = EmbeddedMesh(cell_f, subd_color)
                
        sub_facet_f = df.MeshFunction('size_t', subdomain, subdomain.topology().dim()-1, 0)
        subdomains[subd_color] = sub_facet_f
                
        # We paint boundary with subdomain color for this is the
        # right one on exterior iface; it changes only on the interior.
        df.DomainBoundary().mark(sub_facet_f, subd_color)

        _, f2c_loc = subdomain.init(tdim-1, tdim), subdomain.topology()(tdim-1, tdim) 
        # cell -> its facets that are boundary
        lc2lf = defaultdict(list)
        for f in np.where(sub_facet_f.array() == subd_color)[0]:
            [lc2lf[c].append(f) for c in f2c_loc(f)]
        lc2lf_maps[subd_color] = lc2lf
                
        # Invert parent mapping of subdomain mesh
        gc2lc_maps[subd_color] = {
            v: k for k, v in subdomain.parent_entity_map[mesh.id()][tdim].items()
        }

    # No we need to fix coloring of the shared interior boundaries
    facet_f, lookup = interfaces.facet_f, interfaces.lookup
    ilookup = {v: k for k, v in lookup.items()}

    _, f2c = mesh.init(tdim-1, tdim), mesh.topology()(tdim-1, tdim)
    facets = facet_f.array()
    
    keys = iter(sorted(ilookup.keys(), key=len, reverse=True))

    for subd_colors in takewhile(lambda key: len(key) == 2, keys):
        bdry_color = ilookup[subd_colors]
        bdry_facets, = np.where(facets == bdry_color)

        for subd_color in subd_colors:
            # Get the mappings ...
            gc2lc_map, lc2lf_map = gc2lc_maps[subd_color], lc2lf_maps[subd_color]
            # ... and domain we want to color
            subdomain_facets = subdomains[subd_color]
            sub_mesh = subdomain_facets.mesh()
            
            # gf -> gc -> lc -> lf
            #
            for gf in bdry_facets:
                for gc in f2c(gf):
                    # Cell in other domain?
                    if gc not in gc2lc_map:
                        continue
                    
                    lfs = lc2lf_map[gc2lc_map[gc]]
                    # If cell had only one facet we have a match
                    if len(lfs) == 1:
                        lf = lfs[0]
                    # We compare in terms of midpoints
                    else:
                        gm = df.Facet(mesh, gf).midpoint()
                        lfs = [df.Facet(sub_mesh, lfi) for lfi in lfs]
                        lf = min(lfs, key=lambda l: l.midpoint().distance(gm))

                        assert lf.midpoint().distance(gm) < 1E-13

                        lf = lf.index()
                    # Colors        
                    subdomain_facets[lf] = bdry_color
                    
    return cell_f, interfaces, subdomains


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
    for key in sorted(interfaces, key=len):

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

    
    mesh = df.UnitSquareMesh(128, 128)

    #from gmshnics import gUnitSquare
    #mesh, _ = gUnitSquare(0.2)

    color_f, interface, subdomains = partition(mesh, 4, with_subd_meshes=True)

    df.File('foo.pvd') << color_f
    df.File('bar.pvd') << interface.facet_f
    print(interface.lookup)
    for k, v in subdomains.items():
        df.File(f'subd_{k}.pvd') << v
