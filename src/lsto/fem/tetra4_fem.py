import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental.sparse import BCOO
import datetime
import numpy as np

_NONDIAG_ID=jnp.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                      11, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                      23, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                      40, 41, 42, 43, 44, 45, 46, 47, 53, 54,
                      55, 56, 57, 58, 59, 66, 67, 68, 69, 70,
                      71, 79, 80, 81, 82, 83, 92, 93, 94, 95,
                      105, 106, 107, 118, 119, 131])
_DIAG_ID=jnp.arange(12)*13

def jacobian(vertices):
  """
  Compute the Jacobian matrix of tetrahedral elements\\
  vertices : float (n_elem,4,3)

  Return
  ------
  jacobian : float (n_elem,3,3)
  """
  return vertices[:,1:]-vertices[:,0,None]

def dmat(young,poisson):
  out=jnp.array([[1.-poisson,poisson,poisson,0.,0.,0.],
                 [poisson,1.-poisson,poisson,0.,0.,0.],
                 [poisson,poisson,1.-poisson,0.,0.,0.],
                 [0.,0.,0.,(1.-2*poisson)*0.5,0.,0.],
                 [0.,0.,0.,0.,(1.-2*poisson)*0.5,0.],
                 [0.,0.,0.,0.,0.,(1.-2*poisson)*0.5]])
  out=young/(1.+poisson)/(1.-2.*poisson)*out #(6,6)
  return out

def _bmat(vertices):
  """
  vertices : float (n_elem,4,3)
  """
  matdNdab=jnp.array([[-1.0,1.0,0.0,0.0],
                      [-1.0,0.0,1.0,0.0],
                      [-1.0,0.0,0.0,1.0]]) #(3,4)
  matJ=jacobian(vertices) #(n_elem,3,3)
  dNdxy=jnp.linalg.solve(matJ,matdNdab[None,:,:]) #(n_elem,3,4)
  zeros=jnp.zeros((vertices.shape[0],))
  matB=jnp.array([[dNdxy[:,0,0],zeros,zeros,dNdxy[:,0,1],zeros,zeros,dNdxy[:,0,2],zeros,zeros,dNdxy[:,0,3],zeros,zeros],
                  [zeros,dNdxy[:,1,0],zeros,zeros,dNdxy[:,1,1],zeros,zeros,dNdxy[:,1,2],zeros,zeros,dNdxy[:,1,3],zeros],
                  [zeros,zeros,dNdxy[:,2,0],zeros,zeros,dNdxy[:,2,1],zeros,zeros,dNdxy[:,2,2],zeros,zeros,dNdxy[:,2,3]],
                  [zeros,dNdxy[:,2,0],dNdxy[:,1,0],zeros,dNdxy[:,2,1],dNdxy[:,1,1],zeros,dNdxy[:,2,2],dNdxy[:,1,2],zeros,dNdxy[:,2,3],dNdxy[:,1,3]],
                  [dNdxy[:,2,0],zeros,dNdxy[:,0,0],dNdxy[:,2,1],zeros,dNdxy[:,0,1],dNdxy[:,2,2],zeros,dNdxy[:,0,2],dNdxy[:,2,3],zeros,dNdxy[:,0,3]],
                  [dNdxy[:,1,0],dNdxy[:,0,0],zeros,dNdxy[:,1,1],dNdxy[:,0,1],zeros,dNdxy[:,1,2],dNdxy[:,0,2],zeros,dNdxy[:,1,3],dNdxy[:,0,3],zeros],]) #(6,12,n_elem)
  matB=jnp.transpose(matB,axes=(2,0,1)) #(n_elem,6,12)
  return matB

def bmat(vertices):
  """
  vertices : float (n_elem,4,3)
  """
  matdNdab=jnp.array([[-1.0,1.0,0.0,0.0],
                      [-1.0,0.0,1.0,0.0],
                      [-1.0,0.0,0.0,1.0]]) #(3,4)
  matJ=jacobian(vertices) #(n_elem,3,3)
  dNdxy=jnp.linalg.solve(matJ,matdNdab[None,:,:]) #(n_elem,3,4)
  matB=jnp.zeros((vertices.shape[0],72)) #(n_elem,6,12)
  matB=matB.at[:,[ 0, 3, 6, 9,13,16,19,22,26,29,32,35]].set(dNdxy.reshape(-1,12))
  matB=matB.at[:,[50,53,56,59,38,41,44,47,37,40,43,46]].set(dNdxy.reshape(-1,12))
  matB=matB.at[:,[61,64,67,70,60,63,66,69,48,51,54,57]].set(dNdxy.reshape(-1,12))
  return matB.reshape((-1,6,12))

def kelem(vertices,young,poisson,matV):
  """
  Compute the stiffness matrix of tetrahedral elements

  Return
  ------
  matK : float (n_elem,12,12)
  """
  matB=bmat(vertices) #(n_elem,6,12)
  matD=dmat(young,poisson) #(6,6)
  matK=jnp.einsum('nji,jk,nkl->nil', matB, matD, matB)
  matK=matK*matV[:,None,None]/6. #(n_elem,12,12)
  return matK

def melem(rho,matV):
  """
  Compute the uniform diagonal of the lump mass matrix of tetrahedral elements\\
  
  Input
  -----
  matV : float (n_elem,)
  
  Return
  ------
  matM : float (n_elem,)
  """
  matM=rho*matV/24. #(n_elem,)
  return matM

def element_mat(vertices,young,poisson,rho):
  """
  Compute the stiffness and mass matrices of tetrahedral elements\\
  """
  matV=jnp.linalg.det(jacobian(vertices)) #(n_elem,)
  matK=kelem(vertices,young,poisson,matV) #(n_elem,12,12)
  matK=identity1(matK) #(n_elem,12,12)
  matM=melem(rho,matV) #(n_elem,)
  matM=identity2(matM) #(n_elem,)
  matK_trans=matK.reshape(-1,4,3,4,3)#.transpose(0,1,3,2,4).reshape(-1,9) #(nelem*16,9)
  matK_trans=jnp.einsum('aibjc->aijbc', matK_trans).reshape(-1,9) #(nelem*16,9)
  return matK_trans,matM

def gmat_preprocess(connectivity,coordinates,young,poisson,rho):
  """
  Compute the global stiffness matrix (matKg) and mass matrix (matMg) for tetrahedral elements.

  Args:
    connectivity (jnp.ndarray): Element connectivity matrix of shape (nelem,4).
      Each row contains node indices defining a tetrahedral element.
    coordinates (jnp.ndarray): Node coordinates of shape (nnode,3).
    young (float): Young's modulus (elastic modulus).
    poisson (float): Poisson's ratio.
    rho (float): Density (used for mass matrix computation).

  Returns:
    matKg (BCOO): Global stiffness matrix of shape (nnode*3,nnode*3),stored in sparse format.
    matMg (jnp.ndarray): Global mass matrix as a dense array of shape (nnode*3,).
  """
  nnode=coordinates.shape[0]
  vertices=coordinates[connectivity] #(nelem,4,3)
  matK_trans,matM=element_mat(vertices,young,poisson,rho) #(nelem*16,9),(nelem,)
  # Create (i,j) index pairs for stiffness matrix mapping
  ix,iy=jnp.meshgrid(jnp.arange(4),jnp.arange(4),indexing="ij")
  indices=connectivity[:,[ix.flatten(),iy.flatten()]].transpose(0,2,1).reshape(-1,2) # (nelem*16,2)
  u_indices,inv_indices=np.unique(indices,axis=0,return_inverse=True) # (nidx,2),(nelem*16,)
  nidx=u_indices.shape[0]
  matK_trans=jnp.zeros((nidx,9)).at[inv_indices].add(matK_trans).flatten()
  idx_offset=jnp.array([jnp.arange(3).repeat(3),jnp.array([0,1,2]*3)]).T # (9,2)
  idx_full=(u_indices[:,None,:]*3+idx_offset).reshape(-1,2) # (nidx*9,2)

  matKg=BCOO((matK_trans,idx_full),shape=(nnode*3,nnode*3),indices_sorted=True,unique_indices=True) # (nidx*3,nidx*3)
  matMg=jnp.zeros(nnode).at[connectivity.flatten()].add(matM.repeat(4)).repeat(3)
  return matKg,matMg

@custom_vjp
def identity1(x):
  """
  Identity function for JIT compilation
  """
  return x

def identity1_fwd(x):
  """
  Forward pass for identity function
  """
  return x,None

def identity1_bwd(res, g):
  """
  Backward pass for identity function
  """
  print('identity1_bwd called at',datetime.datetime.now())
  return (g,)

identity1.defvjp(identity1_fwd,identity1_bwd)

@custom_vjp
def identity2(x):
  """
  Another identity function for JIT compilation
  """
  return x
def identity2_fwd(x):
  """
  Forward pass for another identity function
  """
  return x,None
def identity2_bwd(res, g):
  """
  Backward pass for another identity function
  """
  print('identity2_bwd called at',datetime.datetime.now())
  return (g,)
identity2.defvjp(identity2_fwd,identity2_bwd)