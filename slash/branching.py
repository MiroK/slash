from collections import defaultdict
import dolfin as df
import numpy as np


def is_loop(mesh):
    '''No bifurcations'''
    assert mesh.topology().dim() == 1
    
    tdim = mesh.topology().dim()
    _, f2c = mesh.init(tdim-1, tdim), mesh.topology()(tdim-1, tdim)
    
    return all(len(f2c(f)) == 2 for f in range(mesh.num_vertices()))


def color_branches(mesh):
    '''Start/end is a terminal'''
    assert mesh.topology().dim() == 1

    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)
    c2v = mesh.topology()(1, 0)

    terminals = {v: set(v2c(v)) for v in range(mesh.num_vertices()) if len(v2c(v)) != 2}
    
    cell_f = df.MeshFunction('size_t', mesh, 1, 0)
    if not terminals:
        cell_f.set_all(1)
        return cell_f, [], [1]

    def next_vertex(c, v, c2v=c2v):
        v0, v1 = c2v(c)
        return v1 if v == v0 else v0

    def next_cell(v, c, v2c=v2c):
        c0, c1 = v2c(v)
        return c1 if c == c0 else c0
    
    values = cell_f.array()
    branch_colors, loop_colors, color = [], [], 0

    exhausted = False
    while not exhausted:
        vertex = max(terminals, key=lambda v: terminals[v])
        vertex_cells = terminals[vertex]

        exhausted = len(vertex_cells) == 0
        # The idea is to walk from vertex following the cell
        while vertex_cells:
            link_cell = vertex_cells.pop()
            v0 = vertex

            branch = [link_cell]
            # v0 --
            while next_vertex(link_cell, v0) not in terminals:
                # -- v0 ==
                v0 = next_vertex(link_cell, v0)
                # Because we have not terminal, ==
                link_cell = next_cell(v0, link_cell)
                branch.append(link_cell)
            # Once we reached the terminal
            v0 = next_vertex(link_cell, v0)

            color += 1
            if v0 == vertex:
                loop_colors.append(color)
            else:
                branch_colors.append(color)
            values[branch] = color
            
            # Preclude leaving from vertex in a look
            link_cell in vertex_cells and vertex_cells.remove(link_cell)
            # If we arrived to some other terminal, we don't want to leave from it by the
            # same way we arrived
            v0 in terminals and link_cell in terminals[v0] and terminals[v0].remove(link_cell)

    return cell_f, branch_colors, loop_colors


def walk_vertices(arg, tag=None):
    '''Walk vertices in a linked way'''
    assert isinstance(arg, df.Mesh) or isinstance(arg, df.MeshFunction)
    # Branch
    if isinstance(arg, df.MeshFunction):
        mesh = arg.mesh()
        assert arg.dim() == 1
    else:
        mesh = arg
        
    assert mesh.topology().dim() == 1 and mesh.geometry().dim() > 1
    # The boring cese
    assert mesh.num_cells() > 1
    
    c2v = mesh.topology()(1, 0)

    cells = walk_cells(arg, tag=tag)
    cell, orient = next(cells)

    vertices = c2v(cell) if orient else reversed(c2v(cell))
    for v in vertices:
        yield v
    
    for cell, orient in cells:
        yield list(c2v(cell) if orient else reversed(c2v(cell)))[-1]
        

def walk_cells(arg, tag=None):
    '''Walk loop mesh of tagged branch of mesh'''
    # We return cell index together with orientation, i.e. True if link
    # is v0, v1 False if link is v1, v0
    assert isinstance(arg, df.Mesh) or isinstance(arg, df.MeshFunction)
    # Branch
    if isinstance(arg, df.MeshFunction):
        mesh = arg.mesh()
        assert arg.dim() == 1
    else:
        mesh = arg
        
    assert mesh.topology().dim() == 1 and mesh.geometry().dim() > 1
    # THe boring cese
    assert mesh.num_cells() > 1
    
    _, v2c = mesh.init(0, 1), mesh.topology()(0, 1)
    c2v = mesh.topology()(1, 0)

    def next_vertex(c, v, c2v=c2v):
        v0, v1 = c2v(c)
        return v1 if v == v0 else v0

    def next_cell(v, c, v2c=v2c):
        c0, c1 = v2c(v)
        return c1 if c == c0 else c0

    if is_loop(mesh):
        link_cell = 0
        # For loop we pick where to start as either of the first cell
        start, v1 = c2v(link_cell)
        # ... and we terminate once we reach the start again
        end = start
    else:
        assert tag is not None
        # If this is a branch we need two end cells/vertices
        cell_indices, = np.where(arg.array() == tag)
        
        vertex2cell = defaultdict(set)
        for cell_idx in cell_indices:
            [vertex2cell[v].add(cell_idx) for v in c2v(cell_idx)]
        # One is a start the other is end
        start, end = [v for v in vertex2cell if len(vertex2cell[v]) == 1]
        
        link_cell, = vertex2cell[start]
        # The linking vertex is not the start
        v1,  = set(c2v(link_cell)) - set((start, ))
        
    yield link_cell, c2v(link_cell)[-1] == v1

    v0 = start
    while next_vertex(link_cell, v0) != end:
        # -- v0 ==
        v0 = next_vertex(link_cell, v0)
        # Because we have not terminal, ==
        link_cell = next_cell(v0, link_cell)

        yield link_cell, c2v(link_cell)[0] == v0
