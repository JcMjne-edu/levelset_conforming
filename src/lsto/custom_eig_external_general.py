from jax import custom_vjp,jit
import jax.numpy as jnp

@custom_vjp
def custom_eigsh_external(A_data,A_indices,B,sol_eigvecs,sol_eigvals,k):
  """
  data : jnp.ndarray (nnz,)
  indices : jnp.ndarray (nnz,)
  nid_valid : int (n,)
  sqrtMg : (n,)

  ---

  v : jnp.ndarray (k,)
  w : jnp.ndarray (n,k)
  """
  v=sol_eigvals[:k]
  w=sol_eigvecs[:,:k]
  #norm_w=jnp.linalg.norm(w,axis=0) #(k2,)
  #w=w/norm_w #(n,k2)
  norm2=(w*B[:,None]*w).sum(axis=0) #(k2,)
  w=w/jnp.sqrt(norm2) #(n,k2)
  return v,w

def custom_eigsh_external_fwd(A_data,A_indices,B,sol_eigvecs,sol_eigvals,k):
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
  #norm_w=jnp.linalg.norm(w,axis=0) #(k2,)
  #w=w/norm_w #(n,k2)
  norm2=(w*B[:,None]*w).sum(axis=0) #(k2,)
  w=w/jnp.sqrt(norm2) #(n,k2)
  B_indices=jnp.arange(B.shape[0]).repeat(2).reshape(-1,2)
  return (v[:k],w[:,:k]),(A_indices,B_indices,v,w,k)

def custom_eigsh_external_bwd(res,g):
  """
  res : (w,indices,shape,k1,k2)
  g : ((k1,),(n,k1))

  v : jnp.ndarray (k2,)
  w : jnp.ndarray (n,k2)
  indices : jnp.ndarray (nnz,2)
  """
  #print('warning: custom_eigsh_external_bwd is in debug mode')
  A_indices,B_indices,v,w,k1=res
  v_trg=v[:k1] # (k1,)
  w_trg=w[:,:k1] # (n,k1)
  difl=jnp.array(v_trg-v[:,None]) # (k2,k1)
  _difl=difl.at[difl==0].set(1.0) # (k2,k1)
  term2=jnp.where(difl==0,0.0,w.T@g[1]/_difl) # (k2,k1)

  #derivative of A
  w2=w@term2 # (n,k1)
  i,j=A_indices.T # (nnz,)
  dv=w_trg[i]*w_trg[j] # (nnz,k1)
  dv=dv@g[0] # (nnz,)
  dw=_core(i,j,w2,w_trg) # (nnz,)
  #dA=dv+dw # (nnz,)
  dA=dv#+dw # (nnz,)

  #derivative of B
  i,j=B_indices.T # (nnz,)
  w_trg2=w_trg[i]**2 # (nnz,k1)
  _dv=-v_trg*w_trg2 # (nnz,k1)
  dv=_dv@g[0] # (nnz,)
  dw=_core(i,j,-w2*v_trg,w_trg) # (nnz,)
  dw2=-0.5*w_trg2@(w_trg*g[1]).sum(axis=0)
  dB=dv+dw+dw2 # (nnz,)
  return dA,None,dB,None,None,None

@jit
def _core(_i,_j,w,w_trg):
  _term=(w[_i]*w_trg[_j]+w[_j]*w_trg[_i])/2. # (_nnz,k1)
  _dw=_term.sum(axis=-1) #(_nnz)
  return _dw

custom_eigsh_external.defvjp(custom_eigsh_external_fwd,custom_eigsh_external_bwd)
