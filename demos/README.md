# Sample output of `bastian_axial.py`
Things take time ...
```
Starting from ((0.0, 0.9999999991578677, 0.0, 0.8106823155657139, 168875),)
Keeping 167822/168875 cells
((0.0, 0.9999999991578677, 0.0, 0.8106823155657139, 168875), (1.5542675962552983e-11, 0.9999999991578675, 5.684341886080802e-14, 0.8106823155657139, 167822))
Keeping 10591/10602 cells
Info    : Meshing 1D...
...
Info    : Done meshing 2D (Wall 340.983s, CPU 340.868s)
Info    : 65349 nodes 115009 elements
Final mesh ((0.36709092822767075, 1.0000000000058051, 0.0003195745801995504, 0.29177786042590054, 96008),)
```

## Mesh size
- This can be adjusted in the code by setting `Mesh.CharacteristicLengthFactor`

## Contour smoothing
- Requires `pyvista` and `networkx`
- Smoothing parameters are `alpha` and `T` final in solving the heat equation
and the contour line temperature that is used to define the smoothed surface

## Partitioning
- Requires [pymetis](https://github.com/inducer/pymetis) installed from source
- We can partition mesh into several (not necessary continguous) pieces and represent
these as subdomains with boundaries marked such that the colors agree between subdomains
that share the boundary

<p align="center">
   <img src="https://github.com/MiroK/slash/blob/master/doc/partition.png">
</p>
