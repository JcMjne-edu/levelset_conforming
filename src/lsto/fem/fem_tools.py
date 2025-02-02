import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import scipy as sp
from lsto.rom.guyan_reduction import guyan_reduction

def MPL_penalty(K:BCOO,nid_constraint):
  """
  K : Stiffness matrix (n,n)\n
  nid_constraint : Constraint node ids\n
  """
  nid_flatten=(nid_constraint.repeat(3).reshape(-1,3)*3+jnp.arange(3)).flatten()
  indices=K.indices
  data=K.data
  
  data_p=jnp.ones(nid_flatten.shape[0])*data.max()*10e4
  indices_p=jnp.array([nid_flatten,nid_flatten]).T

  data=jnp.concatenate((data,data_p)) # (n,)
  indices=jnp.concatenate((indices,indices_p)) # (n,2)

  return BCOO((data,indices),shape=K.shape)

def _guyan_decomp(M:BCOO,active_dim,spc_dim=jnp.zeros(0)):
  """
  M : BCOO matrix (n,n)\n
  active_dim : Active degrees of freedom (m,)\n

  ASSUMPTION: indices are uinque
  """
  data=M.data
  M_indices=M.indices # (nnz,2)
  n=M.shape[0]
  inactive_dim=jnp.setdiff1d(jnp.arange(n),jnp.concatenate([active_dim,spc_dim]),)
  data_ind=jnp.arange(data.shape[0]) # (nnz,)
  M_map_coo=sp.sparse.coo_matrix((data_ind,(M_indices[:,0],M_indices[:,1])),shape=(n,n))
  M_csc=sp.sparse.csc_matrix(M_map_coo)
  A_csc=M_csc[active_dim][:,active_dim] # (m,m)
  B_csc=M_csc[inactive_dim][:,active_dim] # (n-m,m)
  C_csc=M_csc[inactive_dim][:,inactive_dim] # (n-m,n-m)

  A_coo=A_csc.tocoo(); A_ind=jnp.array([A_coo.row,A_coo.col]).T
  A_bcoo=BCOO((data[A_coo.data],A_ind),shape=A_coo.shape)
  B_coo=B_csc.tocoo(); B_ind=jnp.array([B_coo.row,B_coo.col]).T
  B_bcoo=BCOO((data[B_coo.data],B_ind),shape=B_coo.shape)
  C_coo=C_csc.tocoo(); C_ind=jnp.array([C_coo.row,C_coo.col]).T
  C_bcoo=BCOO((data[C_coo.data],C_ind),shape=C_coo.shape)
  return A_bcoo,B_bcoo,C_bcoo

def reduction_K(K,active_dim,spc_dim):
  A,B,C=_guyan_decomp(K,active_dim,spc_dim)
  K_rom=guyan_reduction(A.data,A.indices,B.data,B.indices,C.data,C.indices,*B.shape)
  return K_rom
