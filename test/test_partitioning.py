from slash.partition import partition
from itertools import chain
import dolfin as df
import numpy as np
import pytest


@pytest.fixture
def mesh_2d():
    return df.UnitSquareMesh(15, 30)


def test_lookup(mesh_2d, nsubs=3):
    cell_f, interface = partition(mesh_2d, nsubs=nsubs, with_subd_meshes=False)

    # Sanity
    assert cell_f.dim() == mesh_2d.topology().dim()
    assert interface.facet_f.dim() == mesh_2d.topology().dim() - 1

    lookup = interface.lookup
    # Every key in interface lookup must be a facet color
    facet_colors = set(interface.facet_f.array()) - set((0, ))
    assert facet_colors == lookup.keys()

    # Every value in the lookup values must be an okay color
    cell_colors = set(cell_f.array())
    assert cell_colors == set(sum(lookup.values(), ()))


def test_exterior_boundary(mesh_2d, nsubs=3):
    # In lookup we can get boundaries
    facet_f = df.MeshFunction('size_t', mesh_2d, mesh_2d.topology().dim()-1, 0)
    df.DomainBoundary().mark(facet_f, 1)
    bdry_facets0 = set([f.index() for f in df.SubsetIterator(facet_f, 1)])

    cell_f, interface = partition(mesh_2d, nsubs=nsubs, with_subd_meshes=False)    
    facet_f, lookup = interface.facet_f, interface.lookup
    facets = facet_f.array()
    
    # Our way
    bdry_facets = set()
    for key in lookup:
        if len(lookup[key]) == 1:
            color, = lookup[key]
            bdry_facets.update(np.where(facets == color)[0])

    assert bdry_facets0 == bdry_facets


def test_interior_boundary(mesh_2d, nsubs=3):

    cell_f, interface = partition(mesh_2d, nsubs=nsubs, with_subd_meshes=False)    
    facet_f, lookup = interface.facet_f, interface.lookup
    facets = facet_f.array()
    
    # If we hide the exterior we get all interior in the lookup
    df.DomainBoundary().mark(facet_f, 0)
    colors0 = set(facets) - set((0, ))
    assert colors0 == set(k for k, v in lookup.items() if len(v) == 2)

    for fcolor in colors0:
        my_iface_facets, = np.where(facets == fcolor)
        # Compute it from subdomains
        iface_facets = find_interface(cell_f, lookup[fcolor])

        assert set(my_iface_facets) == set(iface_facets)


def find_interface(subdomains, colors):
    c0, c1 = colors

    mesh = subdomains.mesh()
    tdim = mesh.topology().dim()
    fdim = tdim - 1

    mesh.init(tdim, tdim-1)
    _, f2c = mesh.init(tdim-1, tdim), mesh.topology()(tdim-1, tdim)

    interface = []
    for cell in chain(df.SubsetIterator(subdomains, c0), df.SubsetIterator(subdomains, c1)):
        for f in cell.entities(fdim):
            try:
                cell0, cell1 = f2c(f)
            except ValueError:
                continue
                
            if subdomains[cell0] == c0 and subdomains[cell1] == c1:
                interface.append(f)
            if subdomains[cell0] == c1 and subdomains[cell1] == c0:
                interface.append(f)

    return interface
                
            
def test_subdomains_volume(mesh_2d, nsubs=3):

    cell_f, interface, subdomains = partition(mesh_2d, nsubs=nsubs, with_subd_meshes=True)    
    # Match integrals
    f = df.Expression('(x[0]+x[1])*(x[0]+x[1])', degree=2)
    dX = df.Measure('dx', domain=cell_f.mesh(), subdomain_data=cell_f)

    for color in subdomains:
        ref = df.assemble(f*dX(color))
        subd = df.assemble(f*df.dx(domain=subdomains[color].mesh()))
        assert abs(ref-subd) < 1E-13


def test_subdomains_facet(mesh_2d, nsubs=3):
    cell_f, interface, subdomains = partition(mesh_2d, nsubs=nsubs, with_subd_meshes=True)    
    # Match integrals
    f = df.Expression('(x[0]+x[1])*(x[0]+x[1])', degree=2)
    dXS = df.Measure('dS', domain=cell_f.mesh(), subdomain_data=interface.facet_f)
    dXs = df.Measure('ds', domain=cell_f.mesh(), subdomain_data=interface.facet_f)    

    lookup = interface.lookup
    for color in lookup:
        sub_ids = lookup[color]
        # Connected to just one means look at exterior integral
        if len(sub_ids) == 1:
            ref_measure = dXs
        else:
            ref_measure = dXS

        ref = df.assemble(f*ref_measure(color))

        # On subdomains it is always ds ...
        for sub_id in sub_ids:
            # ... wrt to lookup up subdomain
            measure = df.ds(domain=subdomains[sub_id].mesh(), subdomain_data=subdomains[sub_id])
            
            value = df.assemble(f*measure(color)) 
            assert abs(ref-value) < 1E-13
