#Differentiable Guyan Reduction
from jax import custom_vjp
import jax.numpy as jnp
import scipy as sp
from sksparse.cholmod import cholesky
from jax import jit
import guyan_reduction_tool

@custom_vjp
def guyan_reduction(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n):
  """
  A : (n,n) BCOO format
  B : (m,n) BCOO format
  C : (m,m) BCOO format
  """
  K1,_=_guyan_reduction_core(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n)
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
  grad_C=jnp.array(guyan_reduction_tool.get_grad_C(g,C_indices,invC_B))
  return grad_A,None,grad_B,None,grad_C,None,None,None

guyan_reduction.defvjp(guyan_reduction_fwd,guyan_reduction_bwd)

def _guyan_reduction_core(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n):
  a=sp.sparse.coo_array((A_data,(A_indices[:,0],A_indices[:,1])),shape=(n,n)).toarray()
  C_coo=sp.sparse.coo_array((C_data,(C_indices[:,0],C_indices[:,1])),shape=(m,m))
  C_csc=sp.sparse.csc_matrix(C_coo)
  B_coo=sp.sparse.coo_array((B_data,(B_indices[:,0],B_indices[:,1])),shape=(m,n))
  B_csr=B_coo.tocsr()
  factor=cholesky(C_csc)
  invC_B=jnp.array(factor(B_coo.toarray())) #(m,n)
  K1=jnp.asarray(a-B_csr.T@invC_B) #(n,n)
  return K1,invC_B
