import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax import jit, custom_vjp
from jax.experimental.sparse import BCOO
import numpy as np
import logging

#logging.basicConfig(filename='lsto.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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

def bmat(vertices):
  """
  vertices : float (n_elem,4,3)
  """
  matdNdab=jnp.array([[-1.0,1.0,0.0,0.0],
                      [-1.0,0.0,1.0,0.0],
                      [-1.0,0.0,0.0,1.0]]) #(3,4)
  matJ=jacobian(vertices) #(n_elem,3,3)
  dNdxy=jnp.linalg.solve(matJ,matdNdab[None,:,:]) #(n_elem,3,4)
  #print('bmat called')
  #dNdxy=jnp.linalg.inv(matJ)@matdNdab #(n_elem,3,4)
  zeros=jnp.zeros((vertices.shape[0],))
  matB=jnp.array([[dNdxy[:,0,0],zeros,zeros,dNdxy[:,0,1],zeros,zeros,dNdxy[:,0,2],zeros,zeros,dNdxy[:,0,3],zeros,zeros],
                  [zeros,dNdxy[:,1,0],zeros,zeros,dNdxy[:,1,1],zeros,zeros,dNdxy[:,1,2],zeros,zeros,dNdxy[:,1,3],zeros],
                  [zeros,zeros,dNdxy[:,2,0],zeros,zeros,dNdxy[:,2,1],zeros,zeros,dNdxy[:,2,2],zeros,zeros,dNdxy[:,2,3]],
                  [zeros,dNdxy[:,2,0],dNdxy[:,1,0],zeros,dNdxy[:,2,1],dNdxy[:,1,1],zeros,dNdxy[:,2,2],dNdxy[:,1,2],zeros,dNdxy[:,2,3],dNdxy[:,1,3]],
                  [dNdxy[:,2,0],zeros,dNdxy[:,0,0],dNdxy[:,2,1],zeros,dNdxy[:,0,1],dNdxy[:,2,2],zeros,dNdxy[:,0,2],dNdxy[:,2,3],zeros,dNdxy[:,0,3]],
                  [dNdxy[:,1,0],dNdxy[:,0,0],zeros,dNdxy[:,1,1],dNdxy[:,0,1],zeros,dNdxy[:,1,2],dNdxy[:,0,2],zeros,dNdxy[:,1,3],dNdxy[:,0,3],zeros],]) #(6,12,n_elem)
  matB=jnp.transpose(matB,axes=(2,0,1)) #(n_elem,6,12)
  return matB

def kelem(vertices,young,poisson,matV):
  """
  Compute the stiffness matrix of tetrahedral elements

  Return
  ------
  matK : float (n_elem,12,12)
  """
  matB=bmat(vertices) #(n_elem,6,12)
  matD=dmat(young,poisson) #(6,6)
  matBt=jnp.transpose(matB,axes=(0,2,1)) #(n_elem,12,6)
  matK=matBt@matD@matB #(n_elem,12,12)
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

#@jit
def element_mat(vertices,young,poisson,rho):
  """
  Compute the stiffness and mass matrices of tetrahedral elements\\
  """
  matV=jnp.linalg.det(jacobian(vertices)) #(n_elem,)
  matK=kelem(vertices,young,poisson,matV) #(n_elem,12,12)
  matM=melem(rho,matV) #(n_elem,)
  return matK,matM

#@jit
def _global_mat_full_preprocess(vertices,connectivity,young,poisson,rho):
  matK,matM=element_mat(vertices,young,poisson,rho) #(n_elem,12,12),(n_elem,)
  indice_k=iK(connectivity) #(n_elem*144,2)
  indice_m=iM(connectivity) #(n_elem*12,2)
  return matK,matM,indice_k,indice_m

def get_elem_mat(connectivity,coordinates,young,poisson,rho):
  """
  Compute the stiffness and mass matrices of tetrahedral elements\\
  Input
  ------
  connectivity : int (n_elem,4)
  coordinates : float (n_node,3)

  Return
  ------
  matK : float (n_elem,12,12)
  matM : float (n_elem,)
  """
  vertices=coordinates[connectivity] #(n_elem,4,3)
  matK,matM=element_mat(vertices,young,poisson,rho) #(n_elem,12,12),(n_elem,)
  mass=matM.sum()*4
  return matK,matM,mass

def global_mat_full(connectivity,coordinates,young,poisson,rho):
  """
  Compute the global stiffness and mass matrices\n
  Input
  ------
  connectivity : int (n_elem,4)
  coordinates : float (n_node,3)

  Return
  ------
  matKg : float (n_node*3,n_node*3)
  matMg : float (n_node*3,)
  """
  n_node=coordinates.shape[0]
  vertices=coordinates[connectivity] #(n_elem,4,3)
  matK,matM,indice_k,indice_m=_global_mat_full_preprocess(vertices,connectivity,young,poisson,rho)
  
  ik_diag=indice_k.reshape(-1,144,2)[:,_DIAG_ID].reshape(-1,2)
  dk_diag=matK.reshape(-1,144)[:,_DIAG_ID].reshape(-1)
  
  k_diag_dense_data=BCOO((dk_diag,ik_diag[:,0,None]),shape=(n_node*3,)).todense() #(n_node*3,)
  k_diag_dense_id=jnp.arange(n_node*3).repeat(2).reshape(-1,2) #(n_node*3,2)
  ik_nondiag=jnp.sort(indice_k.reshape(-1,144,2)[:,_NONDIAG_ID].reshape(-1,2),axis=1)
  dk_nondiag=matK.reshape(-1,144)[:,_NONDIAG_ID].reshape(-1) 
  k_triu=BCOO((dk_nondiag,ik_nondiag),shape=(n_node*3,n_node*3)).sum_duplicates(remove_zeros=False)
  kg_data=jnp.concatenate((k_diag_dense_data,k_triu.data,k_triu.data)) # (nnz2,)
  kg_indices=jnp.concatenate((k_diag_dense_id,k_triu.indices,k_triu.indices[:,::-1])) # (nnz2,2)
  matKg=BCOO((kg_data,kg_indices),shape=(n_node*3,n_node*3))
  matMg=BCOO((matM.flatten().repeat(12),indice_m),shape=(n_node*3,)).todense()
  mass=matM.sum()*4

  return matKg,matMg,mass

@jit
def indice_base(connect):
  """
  connect : (4,)\\
  out : (144,2)
  """
  nc=connect.shape[0] # 4
  x1,x2=jnp.meshgrid(connect*3,connect*3) #(nc,nc)
  y1,y2=jnp.meshgrid(jnp.arange(3),jnp.arange(3)) #(3,3)
  y1=y1.flatten()[None,None,:]; y2=y2.flatten()[None,None,:] #(1,1,9)
  z1=x1[:,:,None]+y1; z2=x2[:,:,None]+y2 #(nc,nc,9)
  z1=z1.reshape(nc,nc,3,3).transpose(0,2,1,3).flatten()
  z2=z2.reshape(nc,nc,3,3).transpose(0,2,1,3).flatten()
  out=jnp.array([z2,z1]).T #(nc^2*9,2)
  return out

#@jit
def iK(connectivity):
  """
  connectivities: (num_elements, nc)\\
  out: (num_elements * (3*nc)**2, 2)
  """
  indices=jax.vmap(indice_base)(connectivity) #(num_elements, (3*nc)**2, 2)
  out=indices.reshape(-1, 2) #(num_elements * (3*nc)**2, 2)
  return out

#@jit
def iM(connectivity):
  """
  connectivityies : (num_elements,4)\\  
  out : (num_elements*12,2)
  """
  connect=(connectivity*3).repeat(3).reshape(-1,3) #(num_elements*nc,3)
  fltr=jnp.arange(3)
  out=(connect+fltr).flatten()[:,None] #(num_elements*nc*3,1)
  return out

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
  matK,matM=element_mat(vertices,young,poisson,rho) #(nelem,12,12),(nelem,)
  matK_trans=matK.reshape(-1,4,3,4,3).transpose(0,1,3,2,4).reshape(-1,9) #(nelem*16,9)
  # Create (i,j) index pairs for stiffness matrix mapping
  ix,iy=jnp.meshgrid(jnp.arange(4),jnp.arange(4),indexing="ij")
  indices=connectivity[:,[ix.flatten(),iy.flatten()]].transpose(0,2,1).reshape(-1,2)
  u_indices,inv_indices=jnp.unique(indices,axis=0,return_inverse=True) # (nidx,2),(nelem*16,)
  nidx=u_indices.shape[0]
  matK_trans=jnp.zeros((nidx,9)).at[inv_indices].add(matK_trans).flatten()
  idx_offset=jnp.array([jnp.arange(3).repeat(3),jnp.array([0,1,2]*3)]).T # (9,2)
  idx_full=(u_indices[:,None,:]*3+idx_offset).reshape(-1,2) # (nidx*9,2)

  matKg=BCOO((matK_trans,idx_full),shape=(nnode*3,nnode*3),indices_sorted=True,unique_indices=True) # (nidx*3,nidx*3)
  matMg=jnp.zeros(nnode).at[connectivity.flatten()].add(matM.repeat(4)).repeat(3)
  return matKg,matMg

