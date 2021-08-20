# This is taken from FEniCS_ii to keep the code base free of many dependencies
from slash.make_mesh_cpp import make_mesh
from collections import defaultdict
from itertools import chain
import dolfin as df
import numpy as np


class EmbeddedMesh(df.Mesh):
    '''
    Construct a mesh of marked entities in marking_function.
    The output is the mesh with cell function which inherited the markers. 
    and an antribute `parent_entity_map` which is dict with a map of new 
    mesh vertices to the old ones, and new mesh cells to the old mesh entities.
    Having several maps in the dict is useful for mortaring.
    '''
    def __init__(self, marking_function, markers):
        if not isinstance(markers, (list, tuple)): markers = [markers]
        
        base_mesh = marking_function.mesh()
        assert base_mesh.topology().dim() >= marking_function.dim()
        # Work in serial only (much like submesh)
        assert df.MPI.size(base_mesh.mpi_comm()) == 1

        gdim = base_mesh.geometry().dim()
        tdim = marking_function.dim()
        assert tdim > 0, 'No Embedded mesh from vertices'

        assert markers, markers

        # NOTE: treating submesh as a separate case is done for performance
        # as it seems that pure python as done below is about 2x slower
        # We reuse a lot of Submesh capabilities if marking by cell_f
        if base_mesh.topology().dim() == marking_function.dim():
            # Submesh works only with one marker so we conform
            color_array = marking_function.array()
            color_cells = dict((m, np.where(color_array == m)[0]) for m in markers)

            # So everybody is marked as 1
            one_cell_f = df.MeshFunction('size_t', base_mesh, tdim, 0)
            for cells in color_cells.values(): one_cell_f.array()[cells] = 1
            
            # The Embedded mesh now steals a lot from submesh
            submesh = df.SubMesh(base_mesh, one_cell_f, 1)

            df.Mesh.__init__(self, submesh)

            # The entity mapping attribute;
            # NOTE: At this point there is not reason to use a dict as
            # a lookup table            
            mapping_0 = submesh.data().array('parent_vertex_indices', 0)
            mapping_tdim = submesh.data().array('parent_cell_indices', tdim)

            mesh_key = marking_function.mesh().id()            
            self.parent_entity_map = {mesh_key: {0: dict(enumerate(mapping_0)),
                                                 tdim: dict(enumerate(mapping_tdim))}}
            # Finally it remains to preserve the markers
            f = df.MeshFunction('size_t', self, tdim, 0)
            f_values = f.array()
            if len(markers) > 1:
                old2new = dict(zip(mapping_tdim, range(len(mapping_tdim))))
                for color, old_cells in color_cells.items():
                    new_cells = np.array([old2new[o] for o in old_cells], dtype='uintp')
                    f_values[new_cells] = color
            else:
                f.set_all(markers[0])
            
            self.marking_function = f
            # Declare which tagged cells are found
            self.tagged_cells = set(markers)
            # https://stackoverflow.com/questions/2491819/how-to-return-a-value-from-init-in-python            
            return None  

        # Otherwise the mesh needs to by build from scratch
        _, e2v = (base_mesh.init(tdim, 0), base_mesh.topology()(tdim, 0))
        entity_values = marking_function.array()
        colorings = [np.where(entity_values == tag)[0] for tag in markers]
        # Represent the entities as their vertices
        tagged_entities = np.hstack(colorings)

        tagged_entities_v = np.array([e2v(e) for e in tagged_entities], dtype='uintp')
        # Unique vertices that make them up are vertices of our mesh
        tagged_vertices = np.unique(tagged_entities_v.flatten())
        # Representing the entities in the numbering of the new mesh will
        # give us the cell makeup
        mapping = dict(zip(tagged_vertices, range(len(tagged_vertices))))
        # So these are our new cells
        tagged_entities_v.ravel()[:] = np.fromiter((mapping[v] for v in tagged_entities_v.flat),
                                                   dtype='uintp')
        
        # With acquired data build the mesh
        df.Mesh.__init__(self)
        # Fill
        vertex_coordinates = base_mesh.coordinates()[tagged_vertices]
        make_mesh(coordinates=vertex_coordinates, cells=tagged_entities_v, tdim=tdim, gdim=gdim,
                  mesh=self)

        # The entity mapping attribute
        mesh_key = marking_function.mesh().id()
        self.parent_entity_map = {mesh_key: {0: dict(enumerate(tagged_vertices)),
                                             tdim: dict(enumerate(tagged_entities))}}

        f = df.MeshFunction('size_t', self, tdim, 0)
        # Finally the inherited marking function. We colored sequentially so
        if len(markers) > 1:
            f_ = f.array()            
            offsets = np.cumsum(np.r_[0, list(map(len, colorings))])
            for i, marker in enumerate(markers):
                f_[offsets[i]:offsets[i+1]] = marker
        else:
            f.set_all(markers[0])

        self.marking_function = f
        # Declare which tagged cells are found
        self.tagged_cells = set(markers)


def ContourMesh(contour):
    '''Build 1d mesh takingh successive vertices as cells'''
    # NOTE: for several contours we assume that they don't intersect, 
    # have common vertices (but this is not checked)

    if isinstance(contour, np.ndarray):
        contour = (contour, )
    assert isinstance(contour, (tuple, list))
    assert all(isinstance(c, np.ndarray) for c in contour)
    assert all(np.linalg.norm(c[0] - c[-1]) < 1E-13 for c in contour)

    tdim = 1
    coordinates, cells, offsets = [], [], [0]
    for c in contour:
        _, gdim = c.shape
        
        coordinates_ = c[:-1]
        nvtx = len(coordinates_)
        cells_ = offsets[-1] + np.array([(i, (i+1)%nvtx) for i in range(nvtx)])

        coordinates.append(coordinates_)
        cells.append(cells_)
        offsets.append(offsets[-1] + nvtx)
    # Cast
    coordinates, cells = map(np.row_stack, (coordinates, cells))

    mesh = df.Mesh()
    make_mesh(coordinates=coordinates, cells=cells, tdim=tdim, gdim=gdim,
              mesh=mesh)

    # Color them
    cell_f = df.MeshFunction('size_t', mesh, tdim, 0)
    values = cell_f.array()
    for c, (first, last) in enumerate(zip(offsets[:-1], offsets[1:]), 1):
        values[first:last] = c

    return mesh, cell_f
