import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import scipy as sp
import numpy as np


_nid_tetra=jnp.array([[0,1,14,9],[0,1,8,14],[1,2,14,10],[1,2,8,14],
                      [2,3,14,11],[2,3,8,14],[3,0,14,12],[3,0,8,14],
                      [0,14,4,9],[0,14,12,4],[1,5,14,9],[1,10,14,5],
                      [2,6,14,10],[2,6,11,14],[3,11,7,14],[3,14,7,12],
                      [4,5,9,14],[4,5,14,13],[5,6,10,14],[5,6,14,13],
                      [6,7,11,14],[6,7,14,13],[4,14,7,13],[4,12,7,14]]) #(24,4)
i1=np.array([0,1,2,3,4,5,6,7,
             8,8,8,8,  9,9,9,9,  10,10,10,10,
             11,11,11,11,  12,12,12,12,  13,13,13,13,
             14,14,14,14,14,14,14,14])
i2=np.array([0,1,2,3,4,5,6,7,
             0,1,2,3,  0,1,4,5,  1,2,5,6,
             2,3,6,7,  0,3,4,7,  4,5,6,7,
             0,1,2,3,4,5,6,7])
data=np.array([1.]*8+[0.25]*24+[0.125]*8)
map_aug=sp.sparse.coo_array((data,(i1,i2)),shape=(15,8))
_map_aug=BCOO.from_scipy_sparse(map_aug) # (15,8)

def redivision(phi,connect):
  """
  phi : (n,)
  connect : (m,8)
  coord : (n,3)
  """
  phis=phi[connect] # (m,8)
  phi_ex=phis@_map_aug.T #(m,15)
  return phi_ex.flatten()

def redivision_connect_coord(connect,coord):
  """
  phi : (n,)
  connect : (m,8)
  coord : (n,3)
  """
  vertices=coord[connect] # (m,8,3)
  v_ex=_map_aug@vertices # (m,15,3)
  coord_ex=v_ex.reshape(-1,3)
  connect_ex=jnp.arange(v_ex.shape[0]*v_ex.shape[1]).reshape((v_ex.shape[0],v_ex.shape[1])) # (m,27)
  connect_ex=connect_ex[:,_nid_tetra].reshape(-1,4) # (~,4)
  return connect_ex,coord_ex
