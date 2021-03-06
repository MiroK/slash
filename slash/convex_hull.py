import collections
from itertools import starmap
import numpy as np


Point = collections.namedtuple('Point', ('x', 'y'))


def convex_hull(points):
    "Find the convex hull of a set of points."
    # From numpy to numpy
    ret_type = list
    if isinstance(points, np.ndarray):
        ret_type = np.ndarray
        
        assert points.ndim == 2
    
        npts, gdim = points.shape
        assert gdim == 2
        points = tuple(starmap(Point, points))

    # This is Peter Norvig's algorithm    
    if len(points) <= 3:
        return points
    # Find the two half-hulls and append them, but don't repeat first and last points
    upper = half_hull(sorted(points))
    lower = half_hull(reversed(sorted(points)))
    hull = upper + lower[1:-1]

    if ret_type is np.ndarray:
        return np.array(hull)
    return hull


def half_hull(sorted_points):
    "Return the half-hull from following points in sorted order."
    # Add each point C in order; remove previous point B if A->B-C is not a left turn.
    hull = []
    for C in sorted_points:
        # if A->B->C is not a left turn ...
        while len(hull) >= 2 and turn(hull[-2], hull[-1], C) != 'left':
            hull.pop() # ... then remove B from hull.
        hull.append(C)
    return hull


def turn(A, B, C):
    "Is the turn from A->B->C a 'right', 'left', or 'straight' turn?"
    diff = (B.x - A.x) * (C.y - B.y)  -  (B.y - A.y) * (C.x - B.x) 
    return ('right' if diff < 0 else
            'left'  if diff > 0 else
            'straight')
