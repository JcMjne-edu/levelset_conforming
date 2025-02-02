from jax import custom_vjp,jit
import jax.numpy as jnp
import numpy as np

@custom_vjp
def custom_eigvalsh_external(matKe,matMe,elems_tet,nid_surf,sol_eigvecs,sol_eigvals,k):
  """
  matKe : (ne,12,12)
  matMe : (ne,12)
  elems_tet : (ne,4)

  ---

  v : jnp.ndarray (k,)
  w : jnp.ndarray (n,k)
  """
  v=sol_eigvals[:k]
  w=sol_eigvecs[:,:k]
  return v,w

def custom_eigvalsh_external_fwd(matKe,matMe,elems_tet,nid_surf,sol_eigvecs,sol_eigvals,k):
  """
  data : float (nnz,)
  indices : int (nnz,2)
  nid_valid : int (n_valid,)
  sqrtMg : float (n_valid,)

  ----

  v : jnp.ndarray (k2,)
  w : jnp.ndarray (n,k2)
  """
  v=sol_eigvals #(k2,)
  w=sol_eigvecs #(n,k2)
  
  msk_elem=_get_msk_elem(elems_tet,nid_surf)
  idx1,idx2,idxm=get_matKe_dim(elems_tet[msk_elem])
  return (v[:k],w[:,:k]),(matKe,matMe,msk_elem,idx1,idx2,idxm,v,w,k)

def custom_eigvalsh_external_bwd(res,g):
  """
  res : (w,indices,shape,k1,k2)
  g : ((k1,),(n,k1))

  v : jnp.ndarray (k2,)
  w : jnp.ndarray (n,k2)
  indices : jnp.ndarray (nnz,2)
  """
  matKe,matMe,msk_elem,idx1,idx2,idxm,v,w,k1=res

  v_trg=v[:k1] # (k1,)
  w_trg=w[:,:k1] # (n,k1)
  #difl=jnp.array(v_trg-v[:,None]) # (k2,k1)
  #_difl=difl.at[difl==0].set(1.0) # (k2,k1)
  #term2=jnp.where(difl==0,0.0,w.T@g[1]/_difl) # (k2,k1)

  #derivative of A
  #w2=w@term2 # (n,k1)
  #i,j=A_indices.T # (nnz,)
  dv=w_trg[idx1]*w_trg[idx2] # (nnz,k1)
  print('0')
  dv=dv@g[0] # (nnz,)
  #dw=_core(i,j,w2,w_trg) # (nnz,)
  dA=dv.reshape(-1,12,12)#+dw # (nnz,)

  #derivative of B
  #i,j=B_indices.T # (nnz,)
  w_trg2=w_trg[idxm]**2 # (nnz,k1)
  _dv=-v_trg*w_trg2 # (nnz,k1)
  dv=_dv@g[0] # (nnz,)
  #dw=_core(i,j,-w2*v_trg,w_trg) # (nnz,)
  #dw2=-0.5*w_trg2@(w_trg*g[1]).sum(axis=0)
  dB=dv.reshape(-1,12) #+dw+dw2 # (nnz,12)
  dB=dB.sum(axis=1)
  print('1')
  outKe=jnp.zeros_like(matKe)
  outKe=outKe.at[msk_elem].set(dA)
  outMe=jnp.zeros_like(matMe)
  outMe=outMe.at[msk_elem].set(dB)
  print('2')
  return outKe,outMe,None,None,None,None,None

custom_eigvalsh_external.defvjp(custom_eigvalsh_external_fwd,custom_eigvalsh_external_bwd)

def _get_msk_elem(elem_tet,nid_surf):

  """
  elem_tet : (n,4)
  nid_surf : (m,)
  """
  print('Warning: _get_msk_elem is in debug mode')
  msk_nid=jnp.zeros(elem_tet.max()+1,bool)
  msk_nid=msk_nid.at[nid_surf].set(True)
  msk_elem=msk_nid[elem_tet].any(axis=-1) #(n,)
  
  msk_elem=jnp.ones_like(msk_elem)
  return msk_elem

def get_matKe_dim(connect):
  """
  connect : (m,4)
  """
  dim=np.arange(3).reshape(-1,1).repeat(4,axis=1).T.flatten()
  idx=connect.repeat(3,axis=1)*3+dim #(m,12)
  idx1=idx.repeat(12,axis=1).reshape(-1,12,12) #(m,12,12)
  idx2=idx1.transpose(0,2,1) #(m,12,12)
  return idx1.flatten(),idx2.flatten(),idx.flatten()