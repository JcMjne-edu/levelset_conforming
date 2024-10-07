#Differentiable Guyan Reduction
from jax import custom_vjp
import jax.numpy as jnp
import scipy as sp
import numpy as np
from sksparse.cholmod import cholesky
from jax import jit

@custom_vjp
def guyan_reduction(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n):
  """
  A : (n,n) BCOO format
  B : (m,n) BCOO format
  C : (m,m) BCOO format
  """
  K1,invC_B=_guyan_reduction_core(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n)
  return K1

def guyan_reduction_fwd(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n):
  """
  A : (n,n) BCOO format
  B : (n,m) BCOO format
  C : (m,m) BCOO format
  """
  K1,invC_B=_guyan_reduction_core(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n)
  return K1,(A_indices,B_indices,C_indices,invC_B)

def guyan_reduction_bwd(res,g):
  """
  g : (n,n)

  invC_B : (m,n)
  """
  A_indices,B_indices,C_indices,invC_B=res
  
  grad_A=g[A_indices[:,0],A_indices[:,1]]
  grad_B=-(invC_B@g)*2
  grad_B=grad_B[B_indices[:,0],B_indices[:,1]]
  grad_C=_grad_C(g,invC_B,C_indices)
  return grad_A,None,grad_B,None,grad_C,None,None,None

guyan_reduction.defvjp(guyan_reduction_fwd,guyan_reduction_bwd)

def _guyan_reduction_core(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n):
  a=sp.sparse.coo_array((A_data,(A_indices[:,0],A_indices[:,1])),shape=(n,n)).toarray()
  C_coo=sp.sparse.coo_array((C_data,(C_indices[:,0],C_indices[:,1])),shape=(m,m))
  C_csc=sp.sparse.csc_matrix(C_coo)
  B_coo=sp.sparse.coo_array((B_data,(B_indices[:,0],B_indices[:,1])),shape=(m,n))
  B_csr=B_coo.tocsr()
  #print(C_csc.diagonal().min())
  #print(C_csc.diagonal().max())
  factor=cholesky(C_csc)
  invC_B=jnp.array(factor(B_coo.toarray())) #(m,n)
  K1=jnp.asarray(a-B_csr.T@invC_B) #(n,n)
  return K1,invC_B

def _grad_C(g,invC_B,C_indices,max_nelem=2**29):
  """
  g : (n,n)
  invC_B : (m,n)
  C_indices : (nnz,2)
  """
  n_sep=int(invC_B.shape[1]*C_indices.shape[0]/max_nelem)+1
  sep_size=int(C_indices.shape[0]/n_sep)+1
  start_id=np.arange(0,n_sep)*sep_size
  end_id=start_id+sep_size
  grad_C=jnp.zeros(C_indices.shape[0])
  term1=invC_B@g
  for sid,eid in zip(start_id,end_id):
    #term=_grad_C_core(g,invC_B,C_indices[sid:eid])
    t1=term1[C_indices[sid:eid,0]] #(nnz,n)
    t2=invC_B[C_indices[sid:eid,1]] #(nnz,n)
    #grad_C_partial=(t1*t2).sum(axis=1) #(nnz,)
    grad_C_partial=_dot(t1,t2) #(nnz,)
    #print(sid,eid,grad_C_partial.shape)
    grad_C=grad_C.at[sid:eid].set(grad_C_partial)
  #term1=jnp.asarray(invC_B[C_indices[:,0]]) #(nnz,n)
  #term2=jnp.asarray(invC_B[C_indices[:,1]]) #(nnz,n)
  #grad_C=((term1@g)*term2).sum(axis=1) #(nnz,)
  return grad_C

@jit
def _dot(a,b):
  out=(a*b).sum(axis=1)
  return out