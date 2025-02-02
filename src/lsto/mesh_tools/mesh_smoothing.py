import jax.numpy as jnp
from jax.experimental.sparse import BCOO,BCSR
import scipy as sp

def mapping_laplacian(connect,nn,alpha=0.1):
  """
  Mapping matrix for Laplacian smoothing

  connect : int (ne,3)
  coord : float (nn,3)
  """
  edge=connect[:,[0,1,1,2,2,0]].reshape(-1,2) # (ne*3,2)
  edge=jnp.sort(edge,axis=1)
  edge=jnp.unique(edge,axis=0) # (ne2,2)
  _,n_adj=jnp.unique(edge,return_counts=True) # (nn,)
  val=1./n_adj*alpha #(nn,)
  indices=jnp.concatenate([edge,edge[:,::-1]]) # (ne2*2,2)
  data=val[indices[:,0]] # (ne2*2,)

  indices_diag=jnp.arange(nn)[:,None].repeat(2,axis=1) # (nn,2)
  data_diag=jnp.ones(nn)*(1.-alpha) # (nn,)

  indices=jnp.concatenate([indices,indices_diag]) # (ne2*2+nn,2)
  data=jnp.concatenate([data,data_diag]) # (ne2*2+nn,)
  mapping_coo=sp.sparse.coo_matrix((data,(indices[:,0],indices[:,1])),shape=(nn,nn))
  mapping_csr=mapping_coo.tocsr()
  #mapping_bcsr=BCSR.from_scipy_sparse(mapping_csr)
  return mapping_csr

def mapping_average(connect,nn):
  """
  Mapping matrix for averaging

  connect : int (ne,3)
  coord : float (nn,3)
  """
  edge=connect[:,[0,1,1,2,2,0]].reshape(-1,2) # (ne*3,2)
  edge=jnp.sort(edge,axis=1)
  edge=jnp.unique(edge,axis=0) # (ne2,2)
  _,n_adj=jnp.unique(edge,return_counts=True) # (nn,)
  val=1./n_adj #(nn,)
  indices=jnp.concatenate([edge,edge[:,::-1]]) # (ne2*2,2)
  data=val[indices[:,0]] # (ne2*2,)
  mapping_coo=sp.sparse.coo_matrix((data,(indices[:,0],indices[:,1])),shape=(nn,nn))
  mapping_csr=mapping_coo.tocsr()
  #mapping_bcsr=BCSR.from_scipy_sparse(mapping_csr)
  return mapping_csr

def laplacian_smoothing(connect,coord,alpha=0.1):
  """
  Laplacian smoothing of the mesh

  connect : int (ne,3)
  coord : float (nn,3)
  """
  mapping=mapping_laplacian(connect,coord.shape[0],alpha) # (nn,nn)
  new_coord=mapping@coord # (nn,3)
  return new_coord

def taubin_smoothing(connect,coord,lmd=0.1,n_iter=10):
  """
  Taubin smoothing of the mesh

  connect : int (ne,3)
  coord : float (nn,3)
  """
  a=.9
  mu=1./(a-1./lmd)
  print(lmd,mu)
  mapping_p=mapping_laplacian(connect,coord.shape[0],lmd) # (nn,nn)
  mapping_n=mapping_laplacian(connect,coord.shape[0],mu) # (nn,nn)
  mapping=mapping_n@mapping_p
  for i in range(n_iter):
    mapping=mapping@mapping
  return mapping@coord

def mapping_taubin_smoothing(connect,nn,lmd=0.02,n_iter=10):
  a=.5
  mu=1./(a-1./lmd)
  mapping_p=mapping_laplacian(connect,nn,lmd) # (nn,nn)
  mapping_n=mapping_laplacian(connect,nn,mu) # (nn,nn)
  mapping=mapping_n@mapping_p
  for _ in range(n_iter):
    mapping=mapping@mapping
  mapping_bcsr=BCSR.from_scipy_sparse(mapping)
  return mapping_bcsr

def hc_laplacian_smoothing(connect,coord,n,lmd=0.1,mu=-0.1):
  """
  Laplacian smoothing of the mesh

  connect : int (ne,3)
  coord : float (nn,3)
  """
  mapping_lap=mapping_laplacian(connect,coord.shape[0],lmd) # (nn,nn)
  mapping_avr=mapping_average(connect,coord.shape[0]) # (nn,nn)
  mapping=mu*mapping_avr+mapping_lap-mu*mapping_avr@mapping_lap
  for i in range(n):
    mapping=mapping@mapping
  new_coord=mapping@coord # (nn,3)
  return new_coord