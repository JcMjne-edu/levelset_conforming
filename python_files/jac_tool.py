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

def get_nearest_indices(coord_ls_str,v,num_point=1):
  """
  Find the nearest point in coord_ls_str to v

  coord_ls_str: (n,3) array
  v: (m,3) array
  num_point: int
  """
  d=coord_ls_str-v[:,None] # (m,n,3)
  d=jnp.linalg.norm(d,axis=2) # (m,n)
  indices=jnp.argsort(d,axis=1)[:,:num_point] # (m,num_point)
  indices=jnp.unique(indices.flatten())
  return indices

def get_label_surfElem(connect):
  """
  connect: (n,4) int array
  """
  connect_sorted=jnp.sort(connect,axis=1)
  faces=connect_sorted[:,[0,1,2,0,1,3,0,2,3,1,2,3]].reshape(-1,3) #(4n,3)
  _,inv,counts=jnp.unique(faces,return_inverse=True,return_counts=True,axis=0) 
  metric=counts[inv].reshape(-1,4) #(n,4)
  label=jnp.min(metric,axis=1)==1 #(n,)
  return label

def update_phi(phi,connect,coord,coord_ls_str,threashold=0.1,num_point=1):
  """
  phi: (l,4) float array
  connect: (n,4) int array
  coord: (m,3) float array
  coord_ls_str: (s,3) float array
  """
  vertices=coord[connect]
  metric=get_vol_metric(vertices)
  label=get_label_surfElem(connect)
  metric=metric.at[label].set(0.0)
  elem_trg=jnp.where(metric>threashold)[0]
  v=get_coord_center(vertices[elem_trg])
  indices=get_nearest_indices(coord_ls_str,v,num_point)
  phi_updated=phi.at[indices].set(1.0)
  return phi_updated,elem_trg
  