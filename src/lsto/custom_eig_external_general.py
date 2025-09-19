from jax import custom_vjp
import jax.numpy as jnp
import time

@custom_vjp
def custom_eigsh_external(A_data,A_indices,B,sol_eigvecs,sol_eigvals):#,k):
  """
  data : jnp.ndarray (nnz,)
  indices : jnp.ndarray (nnz,)
  nid_valid : int (n,)
  sqrtMg : (n,)

  ---

  v : jnp.ndarray (k,)
  w : jnp.ndarray (n,k)
  """
  v=sol_eigvals#[:k]
  w=sol_eigvecs#[:,:k]
  #norm_w=jnp.linalg.norm(w,axis=0) #(k2,)
  #w=w/norm_w #(n,k2)
  norm2=(w*B[:,None]*w).sum(axis=0) #(k2,)
  w=w/jnp.sqrt(norm2) #(n,k2)
  return v,w

def custom_eigsh_external_fwd(A_data,A_indices,B,sol_eigvecs,sol_eigvals):#,k):
  """
  data : float (nnz,)
  indices : int (nnz,2)
  nid_valid : int (n_valid,)
  sqrtMg : float (n_valid,)

  ----

  v : eigenvalue jnp.ndarray (k2,)
  w : eigenvector jnp.ndarray (n,k2)
  """
  v=sol_eigvals #(k2,)
  w=sol_eigvecs #(n,k2)
  #norm_w=jnp.linalg.norm(w,axis=0) #(k2,)
  #w=w/norm_w #(n,k2)
  norm2=(w*B[:,None]*w).sum(axis=0) #(k2,)
  w=w/jnp.sqrt(norm2) #(n,k2)
  B_indices=jnp.arange(B.shape[0])
  #return (v[:k],w[:,:k]),(A_indices,B_indices,v,w,k)
  return (v,w),(A_indices,B_indices,v,w)

def custom_eigsh_external_bwd(res,g):
  """
  res : (A_indices, B_indices, v, w)
  g : ((k1,),(n,k1))

  v : jnp.ndarray (kfull,)
  w : jnp.ndarray (n,kfull)
  indices : jnp.ndarray (nnz,2)
  """
  start=time.time()
  if jnp.abs(g[1]).max()==0.0:
    VAL_ONLY=True
  else:
    VAL_ONLY=False
  #print('warning: custom_eigsh_external_bwd is in debug mode')
  #A_indices,B_indices,v,w,k1=res
  mode_valid=((jnp.abs(g[1]).max(axis=0)>0.0)+(jnp.abs(g[0])>0.0))
  A_indices,B_indices,v,w=res
  v_trg=v[mode_valid] # (kvalid,)
  w_trg=w[:,mode_valid] # (n,kvalid)
  w_trg_g=w_trg*g[1][:,mode_valid] # (n,kvalid)
  difl=jnp.array(v_trg-v[:,None]) # (kfull,kvalid)
  _difl=difl.at[difl==0].set(1.0) # (kfull,kvalid)
  term2=jnp.where(difl==0,0.0,w.T@g[1][:,mode_valid]/_difl) # (kfull,kvalid)

  #derivative of A
  w2=w@term2 # (n,kvalid)
  i,j=A_indices.T # (nnz,)
  dv=w_trg[i]*w_trg[j] # (nnz,kvalid)
  dv=dv@g[0][mode_valid] # (nnz,)
  if VAL_ONLY:
    dA=dv
  else:
    dw=_core(i,j,w2,w_trg) # (nnz,)
    dA=dv+dw # (nnz,)
  #derivative of B
  i=B_indices # (nnz,)
  w_trg2=w_trg[i]**2 # (nnz,kvalid)
  _dv=-v_trg*w_trg2 # (nnz,kvalid)
  dv=_dv@g[0][mode_valid] # (nnz,)
  if VAL_ONLY:
    dB=dv
  else:
    dw=_core(i,i,-w2*v_trg,w_trg_g) # (nnz,)
    dw2=-0.5*w_trg2@(w_trg_g).sum(axis=0)
    dB=dv+dw+dw2 # (nnz,)
  end=time.time()
  print(f'custom_eigsh_external_bwd time: {end-start:.3f} sec')
  return dA,None,dB,None,None

def _core(i,j,w,w_trg):
  term1=(w[i]*w_trg[j]).sum(axis=-1) #(_nnz,)
  term2=(w[j]*w_trg[i]).sum(axis=-1) #(_nnz,)
  #_term=(w[_i]*w_trg[_j]+w[_j]*w_trg[_i])/2. # (_nnz,k1)
  #_dw=_term.sum(axis=-1) #(_nnz)
  dw=(term1+term2)/2 #(_nnz)
  return dw

custom_eigsh_external.defvjp(custom_eigsh_external_fwd,custom_eigsh_external_bwd)
