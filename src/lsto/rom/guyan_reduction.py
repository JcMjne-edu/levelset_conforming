#Differentiable Guyan Reduction
from jax import custom_vjp
import jax.numpy as jnp
import scipy as sp
from sksparse.cholmod import cholesky
from jax import jit
from lsto.rom import guyan_reduction_tool
from lsto.fem.elem2global import elem2globalK
import numpy as np
from jax._src.interpreters.batching import BatchTracer

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
  if isinstance(g,BatchTracer):
    g=g.val[0]
  
  grad_A=g[A_indices[:,0],A_indices[:,1]]
  grad_B=-(invC_B@g)*2 #
  grad_B=grad_B[B_indices[:,0],B_indices[:,1]]
  outC=guyan_reduction_tool.get_grad_C(g,C_indices,invC_B)
  grad_C=jnp.zeros(outC.shape)
  grad_C=grad_C.at[:].set(outC)
  return grad_A,None,grad_B,None,grad_C,None,None,None

guyan_reduction.defvjp(guyan_reduction_fwd,guyan_reduction_bwd)

def _guyan_reduction_core(A_data,A_indices,B_data,B_indices,C_data,C_indices,m,n):
  a=sp.sparse.coo_array((A_data,(A_indices[:,0],A_indices[:,1])),shape=(n,n)).toarray()
  C_csc=sp.sparse.csc_matrix((C_data,(C_indices[:,0],C_indices[:,1])),shape=(m,m))
  B_csr=sp.sparse.csr_matrix((B_data,(B_indices[:,0],B_indices[:,1])),shape=(m,n))
  #factor=cholesky(C_csc)
  #invC_B=factor(B_csr.toarray()) #(m,n)
  invC_B=guyan_reduction_tool.solve_cholmod(C_csc,B_csr.toarray())
  K1=jnp.asarray(a-B_csr.T@invC_B) #(n,n)
  return K1,invC_B.copy()

def guyan_reduction_matK_core(matK,elem_tet,dim_active,dim_spc):
  """
  matK : float (n_elem,12,12) element stiffness matrix
  elem_tet : int (n_elem,4)
  """
  assert np.intersect1d(dim_active,dim_spc).shape[0]==0 # Active and spc should be disjoint
  matKg=elem2globalK(matK,elem_tet) #(nK,nK)
  dim_C=np.setdiff1d(np.arange(matKg.shape[0]),np.concatenate([dim_active,dim_spc])) #(nC,)
  matA=matKg[dim_active][:,dim_active] #(nA,nA)
  matB=matKg[dim_C][:,dim_active] #(nC,nA)
  matC=matKg[dim_C][:,dim_C] #(nC,nC)
  #factor=cholesky(matC)
  #invC_B=factor(matB.toarray()) #(nC,nA)
  invC_B=guyan_reduction_tool.solve_cholmod(matC,matB.toarray())
  K1=matA-matB.T@invC_B #(nA,nA)
  return K1,invC_B.copy(),dim_C

@custom_vjp
def guyan_reduction_matK(matK,elem_tet,dim_active,dim_spc,nid_surf):
  """
  matK : float (n_elem,12,12) element stiffness matrix
  elem_tet : int (n_elem,4)
  """
  K1,_,_=guyan_reduction_matK_core(matK,elem_tet,dim_active,dim_spc)
  return K1

def guyan_reduction_matK_fwd(matK,elem_tet,dim_active,dim_spc,nid_surf):
  """
  matK : float (n_elem,12,12) element stiffness matrix
  """
  dim_active=np.sort(dim_active); dim_spc=np.sort(dim_spc)
  K1,invC_B,dim_C=guyan_reduction_matK_core(matK,elem_tet,dim_active,dim_spc)
  return K1,(invC_B,elem_tet,nid_surf,dim_active,dim_C,matK.shape)

def guyan_reduction_matK_bwd(res,g):
  """
  g : (nA,nA)
  invC_B : (nC,nA)
  """
  invC_B,elem_tet,nid_surf,dim_active,dim_C,matK_shape=res
  ndim=(elem_tet.max()+1)*12
  msk_dim_A=np.zeros(ndim,bool)
  msk_dim_A[dim_active]=True
  msk_dim_C=np.zeros(ndim,bool)
  msk_dim_C[dim_C]=True

  msk_elem=_get_msk_elem(elem_tet,nid_surf)
  idx1,idx2=_get_matKe_dim(elem_tet[msk_elem])
  out_grad_trg=np.zeros(idx1.shape[0])
  msk_trg_A=msk_dim_A[idx1]*msk_dim_A[idx2]
  msk_trg_B1=msk_dim_C[idx1]*msk_dim_A[idx2]
  msk_trg_B2=msk_dim_C[idx2]*msk_dim_A[idx1]
  msk_trg_C=msk_dim_C[idx1]*msk_dim_C[idx2]
  map_shrink=np.zeros(ndim,int)
  map_shrink[dim_active]=np.arange(dim_active.shape[0])
  map_shrink[dim_C]=np.arange(dim_C.shape[0])

  if msk_trg_A.any():
    out_grad_trg[msk_trg_A]=g[map_shrink[idx1[msk_trg_A]],map_shrink[idx2[msk_trg_A]]]
  if msk_trg_B1.any():
    out_grad_trg[msk_trg_B1]=-(invC_B[map_shrink[idx1[msk_trg_B1]]]*g[map_shrink[idx2[msk_trg_B1]]]).sum(axis=-1)
    out_grad_trg[msk_trg_B2]=-(invC_B[map_shrink[idx2[msk_trg_B2]]]*g[map_shrink[idx1[msk_trg_B2]]]).sum(axis=-1)
  if msk_trg_C.any():
    indices=np.array([map_shrink[idx1[msk_trg_C]],map_shrink[idx2[msk_trg_C]]]).T
    out_grad_trg[msk_trg_C]=guyan_reduction_tool.get_grad_C(g,indices,invC_B)
  out_grad=jnp.zeros(matK_shape)
  out_grad=out_grad.at[msk_elem].set(out_grad_trg.reshape(-1,12,12))
  #out_grad=jnp.array(out_grad.copy())
  return out_grad,None,None,None,None
guyan_reduction_matK.defvjp(guyan_reduction_matK_fwd,guyan_reduction_matK_bwd)

def _get_msk_elem(elem_tet,nid_surf):

  """
  elem_tet : (n,4)
  nid_surf : (m,)
  """
  #print('warning: _get_msk_elem is in debug mode')
  msk_nid=np.zeros(elem_tet.max()+1,bool)
  msk_nid[nid_surf]=True
  msk_elem=msk_nid[elem_tet].any(axis=-1) #(n,)

  #msk_elem=np.ones_like(msk_elem)
  return msk_elem

def _get_matKe_dim(connect):
  """
  connect : (m,4)
  """
  dim=np.arange(3).reshape(-1,1).repeat(4,axis=1).T.flatten()
  idx=connect.repeat(3,axis=1)*3+dim #(m,12)
  idx1=idx.repeat(12,axis=1).reshape(-1,12,12) #(m,12,12)
  idx2=idx1.transpose(0,2,1) #(m,12,12)
  return idx1.flatten(),idx2.flatten()

_i1=np.array([0,0,0,0,1,1,1,2,2,3])
_i2=np.array([0,1,2,3,1,2,3,2,3,3])
_dim1=np.array([0,0,0,1,1,1,2,2,2])
_dim2=np.array([0,1,2,0,1,2,0,1,2])

def _get_matKe_dim_reduced(connect):
  """
  connect : (m,4)
  """
  i1=connect[:,_i1].flatten() #(m*10,)
  i2=connect[:,_i2].flatten() #(m*10,)
  idx=np.array([i1,i2]).T #(m*10,2)
  msk_inv=idx[:,0]>idx[:,1] #(m*10,)
  idx=np.sort(idx,axis=-1) #(m*10,2)
  u_idx,inv_idx=np.unique(idx,return_inverse=True,axis=0) #(m2,2),(m*10,)

  idx1=((u_idx[:,0].repeat(9)*3).reshape(-1,9)+_dim1).flatten() #(m2*10*6)
  idx2=((u_idx[:,1].repeat(9)*3).reshape(-1,9)+_dim2).flatten() #(m2*10*6)
  
  return idx1,idx2,inv_idx,msk_inv # (m2*10*6,),(m2*10*6,),(m*10,)

@custom_vjp
def guyan_reduction_matK_reduced(matK,elem_tet,dim_active,dim_spc,nid_surf):
  """
  matK : float (n_elem,12,12) element stiffness matrix
  elem_tet : int (n_elem,4)
  """
  K1,_,_=guyan_reduction_matK_core(matK,elem_tet,dim_active,dim_spc)
  return K1

def guyan_reduction_matK_reduced_fwd(matK,elem_tet,dim_active,dim_spc,nid_surf):
  """
  matK : float (n_elem,12,12) element stiffness matrix
  """
  dim_active=np.sort(dim_active); dim_spc=np.sort(dim_spc)
  K1,invC_B,dim_C=guyan_reduction_matK_core(matK,elem_tet,dim_active,dim_spc)
  return K1,(invC_B,elem_tet,nid_surf,dim_active,dim_C,matK.shape)

def guyan_reduction_matK_reduced_bwd(res,g):
  """
  g : (nA,nA)
  invC_B : (nC,nA)
  """
  invC_B,elem_tet,nid_surf,dim_active,dim_C,matK_shape=res
  ndim=(elem_tet.max()+1)*12
  msk_dim_A=np.zeros(ndim,bool)
  msk_dim_A[dim_active]=True
  msk_dim_C=np.zeros(ndim,bool)
  msk_dim_C[dim_C]=True

  msk_elem=_get_msk_elem(elem_tet,nid_surf)
  idx1,idx2,inv_idx,msk_inv=_get_matKe_dim_reduced(elem_tet[msk_elem])
  out_grad_trg=np.zeros(idx1.shape[0])

  msk_trg_A=msk_dim_A[idx1]*msk_dim_A[idx2]
  msk_trg_B1=msk_dim_C[idx1]*msk_dim_A[idx2]
  msk_trg_B2=msk_dim_C[idx2]*msk_dim_A[idx1]
  msk_trg_C=msk_dim_C[idx1]*msk_dim_C[idx2]
  
  map_shrink=np.zeros(ndim,int)
  map_shrink[dim_active]=np.arange(dim_active.shape[0])
  map_shrink[dim_C]=np.arange(dim_C.shape[0])

  if msk_trg_A.any():
    out_grad_trg[msk_trg_A]=g[map_shrink[idx1[msk_trg_A]],map_shrink[idx2[msk_trg_A]]]
  if msk_trg_B1.any():
    out_grad_trg[msk_trg_B1]=-(invC_B[map_shrink[idx1[msk_trg_B1]]]*g[map_shrink[idx2[msk_trg_B1]]]).sum(axis=-1)
    out_grad_trg[msk_trg_B2]=-(invC_B[map_shrink[idx2[msk_trg_B2]]]*g[map_shrink[idx1[msk_trg_B2]]]).sum(axis=-1)
  if msk_trg_C.any():
    indices=np.array([map_shrink[idx1[msk_trg_C]],map_shrink[idx2[msk_trg_C]]]).T
    out_grad_trg[msk_trg_C]=guyan_reduction_tool.get_grad_C(g,indices,invC_B)
  
  _out_grd_elem=out_grad_trg.reshape(-1,9)[inv_idx].reshape(-1,3,3) #(m*10,3,3)
  _out_grd_elem[msk_inv]=_out_grd_elem[msk_inv].transpose(0,2,1) #(m*10,3,3)
  _out_grd_elem=_out_grd_elem.reshape(-1,10,3,3) #(m,10,3,3)
  _out_grd_elem_T=_out_grd_elem[:,[1,2,5,3,6,8]].transpose(0,1,3,2) #(m,6,3,3)
  _out_grd_elem=np.concatenate((_out_grd_elem,_out_grd_elem_T),axis=1) #(m,16,3,3)
  _out_grd_elem=_out_grd_elem[:,[0,1,2,3,10,4,5,6,11,12,7,8,13,14,15,9]] #(m,16,9)
  _out_grd_elem=_out_grd_elem.reshape(-1,4,4,3,3).transpose(0,1,3,2,4).reshape(-1,12,12) #(m,12,12)

  out_grad=jnp.zeros(matK_shape)
  out_grad=out_grad.at[msk_elem].set(_out_grd_elem)
  return out_grad,None,None,None,None

guyan_reduction_matK_reduced.defvjp(guyan_reduction_matK_reduced_fwd,guyan_reduction_matK_reduced_bwd)