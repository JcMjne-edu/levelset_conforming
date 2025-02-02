import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
import numpy as np
from scipy.spatial import KDTree
from lsto.mesh_tools import meshbuilder
from lsto.levelset_conforming_cgal import *

radius = 3.5
dir_name='./levelset_data/medium'
os.makedirs(dir_name,exist_ok=True)

coord_fem=np.load('./FEM_trg/coordinates.npy')
coord_geom=np.load('./FEM_trg/coord_geom.npy')
connect_geom=np.load('./FEM_trg/connect_geom.npy')
length_lattice=np.array([20.,20.,20.])
coords_ls_str,connects_ls_str,nid_const=meshbuilder.meshbuilder(coord_geom,connect_geom,*(length_lattice))
nid_var=np.setdiff1d(np.arange(len(coords_ls_str)),nid_const)
connect_ls_ex,coord_ls_ex=redivision_connect_coord(connects_ls_str,coords_ls_str)
v=coords_ls_str/length_lattice
tree = KDTree(v)
pairs = tree.query_pairs(radius)

pairs=np.array(list(pairs))
dif=v[pairs[:,0]]-v[pairs[:,1]] #(n2,3)
dist=jnp.linalg.norm(dif,axis=-1) #(n2,)
r=dist/radius #(n2,)
weight=jnp.maximum(0,1-r)**4*(4*r+1) #(n2,)

np.save(dir_name+'/weight.npy',weight)
np.save(dir_name+'/indices.npy',pairs)
np.save(dir_name+'/connect_ls_ex.npy',connect_ls_ex)
np.save(dir_name+'/coord_ls_ex.npy',coord_ls_ex)
np.save(dir_name+'/coords_ls_str.npy',coords_ls_str)
np.save(dir_name+'/connects_ls_str.npy',connects_ls_str)
np.save(dir_name+'/nid_const.npy',nid_const)
np.save(dir_name+'/nid_var.npy',nid_var)