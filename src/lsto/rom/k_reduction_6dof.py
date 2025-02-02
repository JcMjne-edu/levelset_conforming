import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import scipy as sp
from sksparse.cholmod import cholesky

@jit
def indice_base_6dof(connect):
  """
  connect : (nc,)\\
  out : (36*nc**2,2)
  """
  nc=connect.shape[0]
  x1,x2=jnp.meshgrid(connect*6,connect*6) #(nc*6,nc*6)
  y1,y2=jnp.meshgrid(jnp.arange(6),jnp.arange(6))
  y1=y1.flatten()[None,None,:]; y2=y2.flatten()[None,None,:]
  z1=x1[:,:,None]+y1; z2=x2[:,:,None]+y2
  z1=z1.reshape(nc,nc,6,6).transpose(0,2,1,3).flatten()
  z2=z2.reshape(nc,nc,6,6).transpose(0,2,1,3).flatten()
  out=jnp.array([z2,z1]).T
  return out

@jit
def iK_6dof(connectivity):
  """
  connectivities: (num_elements, nc)\\
  out: (num_elements * (3*nc)**2, 2)
  """
  vmap_indice_base=jax.vmap(indice_base_6dof)
  indices=vmap_indice_base(connectivity) #(num_elements, (3*nc)**2, 2)
  out=indices.reshape(-1, 2) #(num_elements * (3*nc)**2, 2)
  return out

def _guyan_decomp_csc(M,active_dim,spc_dim):
  """
  M : csc matrix (l,l)\n
  active_dim : Active degrees of freedom (m,)\n

  Return
  --------
  A : CSC matrix (n,n)\n
  B : CSC matrix (m,n)\n
  C : CSC matrix (m,m)
  """
  n=M.shape[0]
  if spc_dim is not None:
    inactive_dim=np.setdiff1d(np.arange(n),np.concatenate([active_dim,spc_dim]))
  else:
    inactive_dim=np.setdiff1d(np.arange(n),active_dim)
  A=M[active_dim][:,active_dim] # (n,n)
  B=M[inactive_dim][:,active_dim] # (m,n)
  C=M[inactive_dim][:,inactive_dim] # (m,m)
  return A,B,C

def _Kg(ke3,ke4,connect3,connect4,nnode):
  """
  ke3 : element stiffness matrices for triangular element (ne3,18,18)\n
  ke4 : element stiffness matrices for tetrahedral element (ne4,24,24)\n
  connect3 : connectivity for triangular element (ne3,3)\n
  connect4 : connectivity for tetrahedral element (ne4,4)\n
  
  Return
  --------
  kg : global stiffness matrix in csc format (nnode*6,nnode*6)
  """
  ik_3=iK_6dof(connect3) #(ne3*(3*3)**2,2)
  ik_4=iK_6dof(connect4) #(ne4*(3*4)**2,2)
  data=np.concatenate([ke3.flatten(),ke4.flatten()]) #(ne3*18*18+ne4*24*24,)
  indices=np.concatenate([ik_3,ik_4]) #(ne3*(3*3)**2+ne4*(3*4)**2,2)
  kg=sp.sparse.coo_matrix((data,(indices[:,0],indices[:,1])),shape=(nnode*6,nnode*6))
  kg=sp.sparse.csc_matrix(kg)
  return kg

def reduction(ke3,ke4,connect3,connect4,nnode,nid_trg,spc_nid):
  """
  Calculate reduced stiffness matrix using Guyan reduction\n

  ke3 : element stiffness matrices for triangular element (ne3,18,18)\n
  ke4 : element stiffness matrices for tetrahedral element (ne4,24,24)\n
  connect3 : connectivity for triangular element (ne3,3)\n
  connect4 : connectivity for tetrahedral element (ne4,4)\n
  nnode : number of nodes\n
  nid_trg : surface node ids\n
  """
  kg=_Kg(ke3,ke4,connect3,connect4,nnode)
  active_dim=(nid_trg.repeat(3).reshape(-1,3)*6+jnp.arange(3)).flatten()
  spc_dim=(spc_nid.repeat(6).reshape(-1,6)*6+jnp.arange(6)).flatten()
  A,B,C=_guyan_decomp_csc(kg,active_dim,spc_dim)
  factor=cholesky(C)
  invC_B=factor(B.toarray())
  kg_rom=A-B.T@invC_B # (m,m)
  return np.array(kg_rom),kg,spc_dim

def reduction_from_global(kg,nid_trg):
  """
  Calculate reduced stiffness matrix using Guyan reduction\n

  nid_trg : surface node ids\n
  """
  active_dim=(nid_trg.repeat(3).reshape(-1,3)*6+jnp.arange(3)).flatten()
  A,B,C=_guyan_decomp_csc(kg,active_dim,None)
  factor=cholesky(C)
  invC_B=factor(B.toarray())
  kg_rom=A-B.T@invC_B # (m,m)
  return np.array(kg_rom)

def reduction_z(ke3,ke4,connect3,connect4,nnode,nid_trg,spc_nid):
  """
  Calculate reduced stiffness matrix using Guyan reduction\n

  ke3 : element stiffness matrices for triangular element (ne3,18,18)\n
  ke4 : element stiffness matrices for tetrahedral element (ne4,24,24)\n
  connect3 : connectivity for triangular element (ne3,3)\n
  connect4 : connectivity for tetrahedral element (ne4,4)\n
  nnode : number of nodes\n
  nid_trg : surface node ids\n
  """
  kg=_Kg(ke3,ke4,connect3,connect4,nnode)
  active_dim=nid_trg*6+2
  spc_dim=(spc_nid.repeat(6).reshape(-1,6)*6+jnp.arange(6)).flatten()
  A,B,C=_guyan_decomp_csc(kg,active_dim,spc_dim)
  factor=cholesky(C)
  invC_B=factor(B.toarray())
  kg_rom=A-B.T@invC_B # (m,m)
  return np.array(kg_rom),kg,spc_dim
