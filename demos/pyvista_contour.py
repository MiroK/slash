from collections import defaultdict
from itertools import chain, groupby
import networkx as nx
from copy import deepcopy
import numpy as np


def color_it(c2v, v2c):
    '''Combined colorings'''
    cell_f, bcolors, lcolors = color_branches(c2v, v2c, color0=0)
    # Don't ruin outside data ...
    c2v = deepcopy(c2v)
    # Update the cell connectivity kicking out those we don't want to work with
    color = 0
    for color in chain(bcolors, lcolors):
        cells, = np.where(cell_f == color)
        for cell in cells:
            del c2v[cell]
    # The vertex connectivity of the cells that defined closed loops
    v2c = imap(c2v)
    # Now ask about proper loops. Note that since c2v is smaller we pass
    # in to loop coloring an array that has the size of the original data.
    # This is simpler then mapping back indices
    new_cell_f = np.zeros_like(cell_f, dtype='uintp')
    new_cell_f, new_lcolors = color_loops(c2v, v2c, color0=color+1, cell_f=new_cell_f)
    # Finally update coloring
    for color in new_lcolors:
        cell_f[new_cell_f == color] = color
        lcolors.append(color)
        
    return cell_f, bcolors, lcolors
    
    
def color_loops(c2v, v2c, color0=0, cell_f=None):
    '''No end'''
    # These are proper loops
    terminals = {v: set(v2c[v]) for v in v2c if len(v2c[v]) != 2}
    # Each cell should get a color based on which loop/branch it belongs to
    # NOTE: we might be coloring using reduced c2v and then len(c2v) is
    # smaller than the original cell function
    if cell_f is None:
        cell_f = color0*np.ones(len(c2v), dtype='uintp')
    
    if terminals:
        return cell_f, []
    
    # The issue is that we can have several loops which the simple topology
    # criterium does not distinguish
    # ++++++++
    # + |--| +
    # + |--| +
    # ++++++++
    g = nx.Graph()
    # So cells are nodes in this grapp; walking connected components is
    # walking nodes
    g.add_edges_from(v2c.values())

    lcolors = []
    for color, cc in enumerate(nx.algorithms.connected_components(g), color0):
        cell_f[list(cc)] = color
        lcolors.append(color)
    return cell_f, lcolors


def color_branches(c2v, v2c, color0=0, cell_f=None):
    '''A branch ends in terminals'''
    ncells = len(c2v)
    # A terminal is a node with 1 or >= 2 cells connected to it
    terminals = {v: set(v2c[v]) for v in v2c if len(v2c[v]) != 2}
    # Each cell should get a color based on which loop/branch it belongs to
    if cell_f is None:
        cell_f = color0*np.ones(ncells, dtype='uintp')

    if not terminals:
        return cell_f, [], []

    def next_vertex(c, v, c2v=c2v):
        v0, v1 = c2v[c]
        return v1 if v == v0 else v0

    def next_cell(v, c, v2c=v2c):
        c0, c1 = v2c[v]
        return c1 if c == c0 else c0
    
    branch_colors, loop_colors, color = [], [], color0

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
            # Think about
            #  /\
            # / \_____   this is one loop and one branch
            #  \/
            if v0 == vertex:
                loop_colors.append(color)
            else:
                branch_colors.append(color)
            cell_f[branch] = color
            
            # Preclude leaving from vertex in a loop
            link_cell in vertex_cells and vertex_cells.remove(link_cell)
            # If we arrived to some other terminal, we don't want to leave from it by the
            # same way we arrived
            v0 in terminals and link_cell in terminals[v0] and terminals[v0].remove(link_cell)

    return cell_f, branch_colors, loop_colors


def walk_vertices(cell_f, tag, c2v, v2c, is_loop):
    '''Walk vertices of cells where cell_f == tag in a linked way'''
    cells = walk_cells(cell_f, tag, c2v, v2c, is_loop)
    cell, orient = next(cells)

    vertices = c2v[cell] if orient else reversed(c2v[cell])
    for v in vertices:
        yield v
    
    for cell, orient in cells:
        yield list(c2v[cell] if orient else reversed(c2v[cell]))[-1]


def imap(mapping):
    '''Invert dict mapping item to collection'''
    inverse = defaultdict(set)
    [inverse[item].add(key) for key in mapping for item in mapping[key]]

    return inverse
        

def walk_cells(cell_f, tag, c2v, v2c, is_loop):
    '''Walk cells where cell_f == tag in a linked way'''
    cell_indices, = np.where(cell_f == tag)
    # Localize to tags
    c2v = {c: c2v[c] for c in cell_indices}
    v2c = imap(c2v)
    # We return cell index together with orientation, i.e. True if link
    # is v0, v1 False if link is v1, v0
    def next_vertex(c, v, c2v=c2v):
        v0, v1 = c2v[c]
        return v1 if v == v0 else v0

    def next_cell(v, c, v2c=v2c):
        c0, c1 = v2c[v]
        return c1 if c == c0 else c0

    if is_loop:
        # Pick first marked cell
        link_cell = cell_indices[0]
        # For loop we pick where to start as either of the first cell
        start, v1 = c2v[link_cell]
        # ... and we terminate once we reach the start again
        end = start
    else:
        # If this is a branch we need two end cells/vertices
        # One is a start the other is end
        start, end = [v for v in v2c if len(v2c[v]) == 1]
        
        link_cell, = v2c[start]
        # The linking vertex is not the start
        v1,  = set(c2v[link_cell]) - set((start, ))
        
    yield link_cell, c2v[link_cell][-1] == v1

    v0 = start
    while next_vertex(link_cell, v0) != end:
        # -- v0 ==
        v0 = next_vertex(link_cell, v0)
        # Because we have not terminal, ==
        link_cell = next_cell(v0, link_cell)

        yield link_cell, c2v[link_cell][0] == v0


def as_linked_points(contour):
    '''Represent contour as collection of linked points'''
    grid = contour.cast_to_unstructured_grid()
    # We allow only on cell type ...
    ctype, = grid.cells_dict
    # ... and that better be an interval
    assert ctype == 3
    
    c2v = grid.cells_dict[ctype].tolist()  # Cell in terms of vertex indices
    c2v = dict(enumerate(c2v))

    v2c = imap(c2v)
    # Color branches and loops
    cell_f, bcolors, lcolors = color_it(c2v, v2c)
    print(f'branches {bcolors} loops {lcolors}')
    pieces = []
    # Walk branches
    for color in bcolors:
        points = list(walk_vertices(cell_f, color, c2v, v2c, is_loop=False))
        pieces.append(points)
    # Walk loops
    for color in lcolors:
        points = list(walk_vertices(cell_f, color, c2v, v2c, is_loop=True))
        pieces.append(points)

    x = 1*grid.points
    # We return physical coordinates and then each piece is a lookup
    # index in the points
    return x, pieces


def only(iterable):
    val, = iterable
    return val

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    import dolfin as df
    import pyvista

    mesh = df.RectangleMesh(df.Point(-1, -1), df.Point(1, 1), 64, 64, 'crossed')
    # Make up data
    V = df.FunctionSpace(mesh, 'CG', 1)
    f = df.interpolate(df.Expression('std::min((x[0]-0.5)*(x[0]-0.5) + x[1]*x[1], (x[0]+0.5)*(x[0]+0.5) + x[1]*x[1])', degree=2), V)

    # Use pyvista to find the contour
    cells = mesh.cells()
    num_cells, vtx_per_cell = cells.shape
    offsets = vtx_per_cell*np.arange(num_cells)
    # vtk.VTK_TRIANGLE is 5
    cell_types = 5*np.ones(num_cells, dtype='uintp')

    x, y = mesh.coordinates().T
    coordinates = np.c_[x, y, np.zeros_like(x)]

    grid = pyvista.UnstructuredGrid({5: cells}, coordinates)
    # Represent function on mesh
    grid.point_arrays['f'] = f.compute_vertex_values()
    # grid.plot(show_edges=True)
    
    # Get contour corresponding to the level set
    # NOTE: for neuron we would be working with just one level set value
    # but this is here to check generality of the linking idea
    contours = grid.contour([0.5, 0.25, 0.125])
    # 0.25 gives us a tricky case
    #  /\  /\
    # /  \/  \    We diagonose it as two loops. These is stricly speaking
    # \  /\  /    one contour. If we need to group them one option is to 
    #  \/  \/     condsider point values so
    # contours.plot(show_edges=True)

    contour_values = contours.point_arrays['f']
    # Eyeball check
    plt.figure()
    
    X, pieces = as_linked_points(contours)
    # We are in plane
    X = X[:, :-1]
    # We want to color the piece of countour by same color if it was
    # part of one level set
    pieces = groupby(pieces, key=lambda x: only(set(np.round(contour_values[x], 8))))
    for cval, pieces_of_contour in pieces:
        line = None
        for k, contour_piece in enumerate(pieces_of_contour, 1):
            x, y = X[contour_piece].T
            line, = plt.plot(x, y, color=line.get_color() if line is not None else None)
        print(f'{cval} has {k} pieces')
    plt.axis('equal')
    plt.show()
