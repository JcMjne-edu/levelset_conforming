from jax.experimental.sparse import BCOO, BCSR
import jax.numpy as jnp
import scipy as sp

def sparse_take2d(a:BCOO,indices):
  """
  a : (n,n) BCOO format
  indices : (m,)
  out : (m,m) BCOO format
  """
  data=a.data #(n,)
  ind=a.indices.flatten() #(2n,)
  size=indices.shape[0]
  fltr_size=max(indices.max(),ind.max())+1
  fltr=-jnp.ones(fltr_size,dtype=int)
  uval,uind=jnp.unique(indices,return_index=True)
  fltr=fltr.at[uval].set(uind)
  ind_valid=fltr[ind].reshape(-1,2) #(n,2)
  msk=(ind_valid==-1).any(axis=1) #(n,)
  out=BCOO((data[~msk],ind_valid[~msk]),shape=(size,size))
  return out

def sparse_division(denom,numer):
  """
  denom : BCOO (n,n)
  numer : float (n,)
  """
  ind_denom=denom.indices # (nnz,2)
  data_denom=denom.data # (nnz,)
  sqrt_numer=jnp.sqrt(numer)
  numer1=sqrt_numer[ind_denom[:,0]]
  numer2=sqrt_numer[ind_denom[:,1]]
  data_denom=data_denom/(numer1*numer2)
  return BCOO((data_denom,ind_denom),shape=denom.shape)

def sparse_take2d_variationl(a:BCOO,indices1,indices2):
  """
  a : (n,n) BCOO format
  indices1 : (m1,)
  indices2 : (m2,)
  out : (m1,m2) BCOO format
  """
  data=a.data #(nd,)
  m1=indices1.shape[0]; m2=indices2.shape[0]
  ind1,ind2=a.indices.T #(nd,), (nd,)
  
  #dim1
  fltr_size=max(indices1.max(),ind1.max())+1
  fltr=-jnp.ones(fltr_size,dtype=int)
  uval,uind=jnp.unique(indices1,return_index=True)
  fltr=fltr.at[uval].set(uind)
  ind_valid1=fltr[ind1] #(nd,)
  msk=(ind_valid1==-1)

  #dim2
  fltr_size=max(indices2.max(),ind2.max())+1
  fltr=-jnp.ones(fltr_size,dtype=int)
  uval,uind=jnp.unique(indices2,return_index=True)
  fltr=fltr.at[uval].set(uind)
  ind_valid2=fltr[ind2] #(nd,)
  msk=(ind_valid2==-1)+msk

  ind_valid=jnp.array([ind_valid1,ind_valid2]).T #(n,2)
  out=BCOO((data[~msk],ind_valid[~msk]),shape=(m1,m2))
  return out

def convertBCOO2BCSR(bcoo):
  data=bcoo.data; indices=bcoo.indices; shape=bcoo.shape
  coo=sp.sparse.coo_array((data,(indices[:,0],indices[:,1])),shape=shape)
  csr=coo.tocsr()
  return BCSR.from_scipy_sparse(csr)

def convertBCOO2np(bcoo):
  data=bcoo.data; indices=bcoo.indices; shape=bcoo.shape
  coo=sp.sparse.coo_array((data,(indices[:,0],indices[:,1])),shape=shape)
  return coo.toarray()

def bcoo2csr(bcoo):
  data=bcoo.data; indices=bcoo.indices
  coo=sp.sparse.coo_matrix((data, (indices[:, 0], indices[:, 1])), shape=bcoo.shape)
  return coo.tocsr()