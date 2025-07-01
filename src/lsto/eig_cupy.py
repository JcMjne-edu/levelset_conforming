import cupyx
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import eigsh
import scipy as sp
import scipy.sparse
import numpy as np


def eig_solve_cupy(kmat,mmat,nmodes):
  """
  Compute the eigenvalues and eigenvectors of the generalized eigenvalue problem.
  kmat: sp.sparse.csc_matrix (n,n)
  mmat: np.array (n,)
  nmodes: int
  """
  n=kmat.shape[0]
  idx=np.arange(n)
  kmat_cp=csp.csr_matrix(kmat)
  mmat_sqrt=sp.sparse.csr_matrix((1./np.sqrt(mmat),(idx,idx)),shape=(n,n))
  mmat_sqrt=csp.csr_matrix(mmat_sqrt)
  mat=cupyx.scipy.sparse.csc_matrix(mmat_sqrt@kmat_cp@mmat_sqrt)
  val,vec=csp.linalg.eigsh(mat,k=nmodes,which='SA',tol=1e-4) #val: (nmodes,) vec: (n,nmodes)
  vec=mmat_sqrt@vec
  return val,vec