import numpy as np
import scipy as sp

def elem2globalK(matK,elem_tet):
  """
  matK : float (n_elem,12,12) element stiffness matrix
  elem_tet : int (n_elem,4)
  """
  ndim=(elem_tet.max()+1)*3
  idx1,idx2=_get_matKe_dim(elem_tet)
  matKg=sp.sparse.csc_matrix((matK.flatten(),(idx1,idx2)),shape=(ndim,ndim))
  return matKg

def _get_matKe_dim(connect):
  """
  connect : (m,4)
  """
  dim=np.arange(3).reshape(-1,1).repeat(4,axis=1).T.flatten()
  idx=connect.repeat(3,axis=1)*3+dim #(m,12)
  idx1=idx.repeat(12,axis=1).reshape(-1,12,12) #(m,12,12)
  idx2=idx1.transpose(0,2,1) #(m,12,12)
  return idx1.flatten(),idx2.flatten()