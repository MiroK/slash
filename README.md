# SLASH - SLICE REMESH

Slash implements a "wish-full" thinking idea of getting a new mesh, ideally of
higher quality, of a geometry discretized by an existing mesh. The idea is to
use the mesh to redefine the geometry and then rely on gmsh

## Dependencies
In addition to the `FEniCS` stack (things are tested for version `2019.1.0`) also
`networkx` and `gmsh` are required.

## Installation
Put the directory on `PYTHONPATH`, e.g. `source setup.rc` for this bash session

