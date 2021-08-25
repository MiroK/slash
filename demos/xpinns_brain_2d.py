from slash.partition import partition
import dolfin as df

mesh = df.Mesh('2d_brain_mesh.xml')

# We want to partition mesh in to nsubs pieces. What we get
# back is a cell function marking colored subdomains into which the
# mesh was partitioned. We also get a representation of the interface
# between the subdomains. Interface between each 2 subdomains or true
# exterior interface (wrt to mesh) receive unique colors and there is
# lookup table so that we can see which color belongs to which (one
# or two) subdomains
cell_f, interface, subdomains = partition(mesh, nsubs=5, with_subd_meshes=True)
# The third argument is a dictionary subodmain color -> facet function
# where the facet function is defied on the subdomain and its boundaries
# are colored as inherited from the `interface`

df.File('global_cell_f.pvd') << cell_f
df.File('global_facet_f.pvd') << interface.facet_f

for color, sub in subdomains.items():
    df.File(f'subd_{color}_facet_f.pvd') << sub
