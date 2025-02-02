import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
from jax import config
config.update("jax_enable_x64", True)
from lsto.custom_eigval_general import custom_eigvalsh_external
from lsto.custom_eig_external_general import custom_eigsh_external
from lsto.fem.tetra4_fem import get_elem_mat,global_mat_full
import jax.numpy as jnp

def func1(coord_fem,elems_tet,young,poisson,rho,nid_surf_tet,sol_eigvecs,sol_eigvals,num_mode_trg,v_trg):
  matK,matM,_=get_elem_mat(elems_tet,coord_fem,young,poisson,rho)
  v,_=custom_eigvalsh_external(matK,matM,elems_tet,nid_surf_tet,sol_eigvecs,sol_eigvals,num_mode_trg)
  loss_eigval=(jnp.abs(v/v_trg[:num_mode_trg]-1.)).mean()/0.03
  return loss_eigval

def func2(coord_fem,elems_tet,young,poisson,rho,nid_surf_tet,sol_eigvecs,sol_eigvals,num_mode_trg,v_trg):
  matKg,matMg,_=global_mat_full(elems_tet,coord_fem,young,poisson,rho)
  v,_=custom_eigsh_external(matKg.data,matKg.indices,matMg,sol_eigvecs,sol_eigvals,num_mode_trg)
  loss_eigval=(jnp.abs(v/v_trg[:num_mode_trg]-1.)).mean()/0.03
  return loss_eigval