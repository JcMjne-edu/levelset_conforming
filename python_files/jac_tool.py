import jax
import jax.numpy as jnp

def get_coord_center(vertices):
  """
  vertices: (n_elem,4,3)
  """
  return vertices.mean(axis=1) # (n_elem,3)

def _get_elem_vol_single(vertices):
  jac=vertices[1:]-vertices[0,None]
  det=jnp.linalg.det(jac)
  return det

def grad_elem_vol(vertices):
  """
  vertices: (n_elem,4,3)
  """
  return jax.vmap(jax.grad(_get_elem_vol_single))(vertices)

def get_elem_vol(vertices):
  """
  vertices: (n_elem,4,3)
  """
  jac=vertices[:,1:]-vertices[:,0,None]
  det=jnp.linalg.det(jac)
  return det

def get_vol_metric(vertices):
  """
  vertices: (n_elem,4,3)
  """
  jac_vertices=jnp.load('./jac_vertices.npy')
  grad_vol=grad_elem_vol(vertices)
  jac=(jac_vertices*grad_vol).sum(axis=(1,2))
  vol=get_elem_vol(vertices)
  out=jac/vol
  return out

def get_nearest_indices(coord_ls_str,v,num_pints=1):
  """
  Find the nearest point in coord_ls_str to v

  coord_ls_str: (n,3) array
  v: (m,3) array
  num_points: int
  """
  d=coord_ls_str-v[:,None] # (m,n,3)
  d=jnp.linalg.norm(d,axis=2) # (m,n)
  indices=jnp.argsort(d,axis=1)[:,:num_pints] # (m,num_points)
  return indices