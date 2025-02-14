import os
import shutil
from jax import jit
from jax.experimental.sparse import BCSR
from jax.lax import stop_gradient
import jax.numpy as jnp
from lsto.aeroelastic_scaling import aeroelastic_scaling_wt
from lsto.custom_eig_external_general import custom_eigsh_external
from lsto.custom_thread import CustomThread
from lsto.fem.fem_tools import reduction_K
from lsto.fem.tetra4_fem import gmat_preprocess
from lsto.tetgen_tools import make_poly,postprocess_tetgen
from lsto.mesh_tools.mapping_surfacenode_fast import mapping_surfacenode_fast
from lsto.mesh_tools.mesh_utility import points_in_holes
from lsto.nastran_tools import run_nastran_eig,write_base_nastran
from lsto.levelset2stl_mat_tetra import mat_phi2face_tetra
from lsto.levelset_redivision_med import redivision,redivision_connect_coord
from lsto.loss_func import loss_cossim
from lsto.stl_tools import stl_from_connect_and_coord
from lsto.mesh_tools.preprocess import mesh3d_to_coord_and_connect,reconstruct_coordinates,connect2adjoint,cs_rbf_adjoint
from lsto.mesh_tools.mesh_postprocess_jax import mesh_postprocess_jx
from lsto.mesh_tools.improve_aspect_ratio import improve_aspect
from lsto.mesh_tools import meshbuilder,mapping_dist
import lsto.logger_settings
from logging import getLogger
import numpy as np
import scipy as sp

from jax import config
config.update("jax_enable_x64", True)

def detect_hole(connect):
  edge=connect[:,[0,1,1,2,2,0]].reshape(-1,2)
  edge=np.sort(edge,axis=1)
  u,counts=np.unique(edge,axis=0,return_counts=True)
  if (counts!=2).any():
    raise ValueError('Hole detected')

class LSTP_conforming:
  def __init__(self,v_dspace,c_dspace,length_lattice,dmul,v_geom,c_geom,
    target_length=0.2):
    """
    v_dspace: (nv,3) vertices on the design space boundary
    c_dspace: (ne,4) connectivity of the design space
    v_geom: (nvg,3) vertices on the geometry mesh
    c_geom: (neg,3) connectivity of the geometry mesh
    """
    self.coords_ls_str,self.connects_ls_str,self.nid_const=meshbuilder.meshbuilder(v_dspace,c_dspace,*length_lattice)
    self.nid_var=np.setdiff1d(np.arange(len(self.coords_ls_str)),self.nid_const)
    self.connect_ls_ex,self.coord_ls_ex=redivision_connect_coord(self.connects_ls_str,self.coords_ls_str)
    adjoint=connect2adjoint(self.connects_ls_str)
    self.weightrbf=cs_rbf_adjoint(self.coords_ls_str/length_lattice,dmul,adjoint) #(nv,nv)
    _idx_tetra=np.array([0,4,5,7,0,1,3,7,0,5,1,7,1,2,3,6,1,6,3,7,1,5,6,7])
    connect_ls_tetra=self.connects_ls_str[:,_idx_tetra].reshape(-1,4) #(ne,4)
    self.vertices=self.coords_ls_str[connect_ls_tetra] #(ne,4,3)
    self.vertices_ex=self.coord_ls_ex[self.connect_ls_ex] #(ne,4,3)
    self.connect_ls_tetra=np.asarray(connect_ls_tetra)
    self.v_geom=v_geom
    self.c_geom=c_geom
    self.target_length=target_length
    self.logger_main=getLogger('lsto_main')
    self.logger_aux=getLogger('lsto_aux')
    os.makedirs('./tetgen',exist_ok=True)
    os.makedirs('./nastran',exist_ok=True)

  def get_phi0(self,lx,ly,m,val_hole=10.0,n_iter=1):
    phi=init_phi_uniform_xy(self.coords_ls_str,self.nid_const,self.weightrbf,lx,ly,m,val_hole,n_iter)
    return phi
  
  def get_phi0_shell(self,n_msk=1):
    phi=init_phi_shell(self.coords_ls_str,self.nid_const,self.weightrbf,n_msk)
    return phi
  
  def set_config(self,young,poisson,rho,nastran_exe_path=None,num_mode_trg=6,num_mode_ref=6,mass_scale=1.0):
    self.young=young
    self.poisson=poisson
    self.rho=rho*mass_scale
    self.nastran_exe_path=nastran_exe_path
    self.mass_scale=mass_scale
    self.num_mode_trg=num_mode_trg
    self.num_mode_ref=num_mode_ref
    write_base_nastran(num_mode_ref,young,poisson,self.rho)

  def load_phi(self,phi):
    self.phi=phi
    _phi=self.weightrbf@phi
    _phi=_phi.at[self.nid_const].set(jnp.clip(_phi[self.nid_const],a_max=-0.1))
    _phi_ex=redivision(_phi,self.connects_ls_str)
    numerator,denominator,offset=mat_phi2face_tetra(np.asarray(stop_gradient(_phi_ex)),self.connect_ls_ex)
    mesh3d=((numerator@_phi_ex)@self.vertices_ex[offset])/(denominator@_phi_ex)[:,:,None] 
    self.mesh3d=mesh3d
    for rf in [10,9,8,7,6]:
      try:
        coords_index,connects_ls=mesh3d_to_coord_and_connect(stop_gradient(mesh3d),round_f=rf)
        detect_hole(connects_ls)
        break
      except ValueError:
        continue
    detect_hole(connects_ls)
    coords_ls=mesh3d.reshape(-1,3)[coords_index]
    self.coords_ls_raw=stop_gradient(coords_ls); self.connects_ls_raw=connects_ls
    stl_from_connect_and_coord(self.connects_ls_raw,self.coords_ls_raw).save('./stl/check_raw.stl')
    connects_ls=improve_aspect(np.array(connects_ls),np.array(stop_gradient(coords_ls)))
    coords_ls,connects_ls=mesh_postprocess_jx(coords_ls,connects_ls,thresh_l=self.target_length,thresh_v=1e-1)
    self.coords_hole,connects_ls,coords_ls=points_in_holes(coords_ls,connects_ls)
    coords_closed=np.concatenate([stop_gradient(coords_ls),self.v_geom])
    connects_closed=np.concatenate([connects_ls,self.c_geom+coords_ls.shape[0]])
    connects_marker=np.ones(connects_closed.shape[0])
    connects_marker[connects_ls.shape[0]:]=2
    self.coords_closed=coords_closed; self.connects_closed=connects_closed; self.connects_marker=connects_marker
    self.coords_ls=stop_gradient(coords_ls); self.connects_ls=connects_ls
    stl_from_connect_and_coord(self.connects_ls,self.coords_ls).save('./stl/check.stl')
    self.nid_surf_geom=np.unique(self.c_geom)+self.coords_ls.shape[0]
    return coords_closed,connects_closed,coords_ls,connects_ls
  
  def preprocess(self,phi,eigenanalysis=True):
    coords_closed,connects_closed,coords_ls,connects_ls=self.load_phi(phi)
    self.logger_aux.info('load phi finished')
    nodes_tet,elems_tet,faces_tet=_tetgen_run(coords_closed,connects_closed,self.coords_hole,self.connects_marker)
    self.nodes_tet=nodes_tet
    self.elems_tet=elems_tet
    self.faces_tet=faces_tet
    if eigenanalysis:
      self.eig_thread=CustomThread(target=run_nastran_eig,args=(elems_tet,nodes_tet,self.nastran_exe_path))
      self.eig_thread.start()
    check_vol(elems_tet,nodes_tet)
    #mat_weight,nid_valid_tet,self.nid_surf_tet=mapping_surfacenode_full(stop_gradient(connects_ls),stop_gradient(coords_ls),coords_closed,elems_tet,nodes_tet,faces_tet)
    mat_weight,self.nid_surf_tet=mapping_surfacenode_fast(stop_gradient(connects_ls),stop_gradient(coords_ls),nodes_tet,faces_tet)
    
    self.set_target()
    self.dim_spc=_nid2dim_3d(jnp.where(nodes_tet[:,1]==0.0)[0])
    coord_fem=reconstruct_coordinates(coords_ls,jnp.array(self.nodes_tet),self.nid_surf_tet,mat_weight)
    coord_fem=coord_fem+0.0*phi.sum()
    #matKg,matMg,mass=global_mat_full(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    matKg,matMg=gmat_preprocess(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    #matK,matM,mass=get_elem_mat(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    #return matK,matM,mass
    self.matKg=matKg; self.matMg=matMg
    return matKg,matMg
  
  def main_eigval(self,phi):
    matK,matM,_=self.preprocess(phi)
    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matK.data,matK.indices,matM,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    #v,w=custom_eigvalsh_external(matK,matM,self.elems_tet,self.nid_surf_tet,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.sol_eigvecs=sol_eigvecs
    self.sol_eigvals=sol_eigvals
    loss_eigval=(jnp.abs(v/self.v_trg[:self.num_mode_trg]-1.)).mean()/0.03
    self.v=v; self.w=w
    print(f'loss : {stop_gradient(loss_eigval):.3f} {np.round(stop_gradient(v),3)}')
    return loss_eigval
  
  def main_eigval_prop(self,phi):
    matK,matM,_=self.preprocess(phi)
    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matK.data,matK.indices,matM,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    #v,w=custom_eigvalsh_external(matK,matM,self.elems_tet,self.nid_surf_tet,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.sol_eigvecs=sol_eigvecs
    self.sol_eigvals=sol_eigvals
    v_trg=self.v_trg[cosine_similarity(sol_eigvecs[self.dim_active_rom],self.w_trg)]
    loss_eigval=((v/v[0]/v_trg[:self.num_mode_trg]*v_trg[0]-1.)[1:]**2).mean()*1e4
    self.v=v; self.w=w
    print(f'loss : {stop_gradient(loss_eigval):.3f}')
    print(f'{np.round(stop_gradient(v/v[0]),2)[1:]}')
    print(f'{np.round(stop_gradient(v_trg/v_trg[0]),2)[1:]}')
    print(f'({stop_gradient(v[0])})')
    #print(f'{self.v_trg[:self.num_mode_trg]/self.v_trg[0]} ({self.v_trg[0]})')
    return loss_eigval
  
  def main_eigvec(self,phi):
    matKg,matMg,_=self.preprocess(phi)
    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matKg.data,matKg.indices,matMg,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.sol_eigvecs=sol_eigvecs; self.sol_eigvals=sol_eigvals
    self.v=v; self.w=w
    self.matKg=matKg; self.matMg=matMg
    loss_eigvec=loss_cossim(w[self.dim_active_full],self.w_trg[:,:self.num_mode_trg])*1e4
    print(f'loss : {stop_gradient(loss_eigvec):.3f} {np.round(stop_gradient(v),3)}')
    return loss_eigvec
  
  def main_mass(self,phi):
    _,_,mass=self.preprocess(phi,eigenanalysis=False)
    loss_mass=jnp.abs(mass/self.mass_trg-1.)/0.03
    print(f'loss : {stop_gradient(loss_mass):.3f} {stop_gradient(mass)/self.mass_scale}/{self.mass_trg/self.mass_scale}')
    return loss_mass
  
  def main_static(self,phi):
    matK,_,_=self.preprocess(phi,eigenanalysis=False)
    #kg_reduced=guyan_reduction_matK(matK,self.elems_tet,self.dim_active_rom,self.dim_spc,self.nid_surf_tet)
    self.matKg=matK
    kg_reduced=reduction_K(matK,self.dim_active_rom,self.dim_spc)
    compliance_reduced=jnp.linalg.inv(kg_reduced)
    loss_static=((compliance_reduced-self.comp_trg)**2).sum()*1e2
    loss_static_scale=stop_gradient(compliance_reduced[self.dim_ref_comp,self.dim_ref_comp])/self.ref_comp
    loss_static_scale=(1.-loss_static_scale)
    self.compliance_reduced=compliance_reduced
    print('loss : ',stop_gradient(loss_static),stop_gradient(loss_static_scale))
    return loss_static
  
  def main_static_cosine(self,phi):
    matK,_,_=self.preprocess(phi,eigenanalysis=False)
    self.matKg=matK
    kg_reduced=reduction_K(matK,self.dim_active_rom,self.dim_spc)
    compliance_reduced=jnp.linalg.inv(kg_reduced)
    loss_static=(1.-(compliance_reduced*self.comp_trg).sum()/jnp.linalg.norm(compliance_reduced)/self.comp_trg_norm)*1e2
    self.compliance_reduced=compliance_reduced
    print('loss : ',stop_gradient(loss_static))
    return loss_static
  
  def main_eigval_static(self,phi):
    matK,matM=self.preprocess(phi)
    self.logger_aux.info('preprocess finished')
    kg_reduced=reduction_K(matK,self.dim_active_rom,self.dim_spc)
    compliance_reduced=jnp.linalg.inv(kg_reduced)
    loss_static=(1.-(compliance_reduced*self.comp_trg).sum()/jnp.linalg.norm(compliance_reduced)/self.comp_trg_norm)*1e2
    self.compliance_reduced=compliance_reduced

    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matK.data,matK.indices,matM,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.logger_aux.info('Nastran calculation finished')
    cossim,aux=cosine_similarity(sol_eigvecs[self.dim_active_rom],self.w_trg)
    v_trg=self.v_trg[cossim]
    loss_eigval=((v/v[0]/v_trg[:self.num_mode_trg]*v_trg[0]-1.)[1:]**2).mean()*1e4
    self.v=v; self.w=w
    
    # Logging
    msg_main =f'\nloss : {stop_gradient(loss_static):.3f}'
    msg_main+=f'\nloss : {stop_gradient(loss_eigval):.3f} ({stop_gradient(v[0]):.3f})'
    fmt='{:.2f}  '*(v.shape[0]-1)
    msg_main+=('\nratio: '+fmt.format(*stop_gradient(v/v[0])[1:]))
    msg_main+=('\n ref : '+fmt.format(*(v_trg/v_trg[0])[1:]))
    fmt='{:.4f}  '*aux.shape[0]
    msg_main+=('\n cos : '+fmt.format(*aux))
    self.logger_main.info(msg_main)

    loss=jnp.asarray([loss_static,loss_eigval])
    return loss,loss
  
  def set_target_raw(self,k_l,k_p,coord_trg,nid_trg_rom,nid_trg_eig,comp_trg,mass,v,w):
    self.k_t,self.k_m,k_f=aeroelastic_scaling_wt(k_l,k_p)
    self.coord_trg_rom=coord_trg[nid_trg_rom]
    self.coord_trg_eig=coord_trg[nid_trg_eig]
    self.mass_trg=mass*self.k_m*self.mass_scale
    self.v_trg=v/self.k_t**2/self.mass_scale
    self.comp_trg=comp_trg*self.k_t**2/self.k_m
    self.comp_trg_norm=jnp.linalg.norm(self.comp_trg)
    self.dim_ref_comp=jnp.argmax(jnp.diag(self.comp_trg))
    self.ref_comp=self.comp_trg[self.dim_ref_comp,self.dim_ref_comp]
    k=w.shape[0]
    self.w_trg=w[:,nid_trg_eig].reshape(k,-1).T #(n_v*3,k)
    
  def set_target(self):
    nid_out_local_all_rom,_=mapping_dist.calc_dist(self.coord_trg_rom,self.nodes_tet[self.nid_surf_geom])
    self.surface_nid_identical_rom=self.nid_surf_geom[nid_out_local_all_rom]
    self.dim_active_rom=_nid2dim_3d(self.surface_nid_identical_rom)
    nid_local_eig,_=mapping_dist.calc_dist(self.coord_trg_eig,self.nodes_tet[self.nid_surf_geom])
    self.nid_tet_eig=self.nid_surf_geom[nid_local_eig]
    self.dim_active_full=_nid2dim_3d(self.nid_tet_eig)

def _tetgen_run(coords,connects,hole=None,marker=None):
  make_poly('./tetgen/temp.poly',coords,connects,hole,marker)
  #subprocess.run(['tetgen', '-q10.0F', 'tetgen/temp.poly'],stdout=subprocess.PIPE)
  out=os.system('tetgen -q5.0 ./tetgen/temp.poly > tetgen_output.log 2>&1')
  if out!=0:
    #logging.error(f'Tetgen failed with code {out}')
    raise ValueError('Tetgen failed')
  nodes_tet,elems_tet,faces_tet,face_marker,_=postprocess_tetgen('./tetgen','temp',1)
  faces_tet=faces_tet[face_marker==1]
  shutil.move('./tetgen/temp.poly','./tetgen/temp_OOD.poly')
  shutil.move('./tetgen/temp.1.node','./tetgen/temp_OOD.1.node')
  shutil.move('./tetgen/temp.1.ele','./tetgen/temp_OOD.1.ele')
  shutil.move('./tetgen/temp.1.face','./tetgen/temp_OOD.1.face')
  return nodes_tet,elems_tet,faces_tet

def init_phi_uniform_xy(coord,nid_const,weightrbf,lx,ly,m,val_hole=10.0,n_iter=1):
  """
  connect : int (n,8)
  coord : float (n,3)
  """
  ix=jnp.round(coord[:,0]/lx).astype(int) #(n,)
  iy=jnp.round(coord[:,1]/ly).astype(int) #(n,)
  isum=ix+iy
  nid_hole=jnp.where((isum%(2*m)==0)*(ix%m==0)*(iy%m==0))[0]
  
  msk_root_nid=(coord[:,1]==coord[:,1].min())
  root_nid=jnp.where(msk_root_nid)[0]

  phi=-jnp.ones(coord.shape[0])*0.5
  phi=phi.at[nid_hole].set(val_hole)
  phi=phi.at[root_nid].set(val_hole)
  phi=phi.at[nid_const].set(-0.5)
  for i in range(n_iter):
    phi=weightrbf@phi
    phi=phi.at[nid_const].set(-1.0)
  phi=jnp.clip(phi,-1.0,1.0)
  return phi

def init_phi_uniform_xyz(coord,nid_const,weightrbf,lx,ly,lz,m,val_hole=10.0):
  """
  connect : int (n,8)
  coord : float (n,3)
  """
  ix=jnp.round(coord[:,0]/lx).astype(int) #(n,)
  iy=jnp.round(coord[:,1]/ly).astype(int) #(n,)
  iz=jnp.round(coord[:,2]/lz).astype(int) #(n,)
  isum=ix+iy+iz
  nid_hole=jnp.where((isum%(2*m)==0)*(ix%m==0)*(iy%m==0)*(iz%m==0))[0]
  
  msk_root_nid=(coord[:,1]==coord[:,1].min())
  root_nid=jnp.where(msk_root_nid)[0]

  phi=-jnp.ones(coord.shape[0])*0.5
  phi=phi.at[nid_hole].set(val_hole)
  phi=phi.at[root_nid].set(val_hole)
  phi=phi.at[nid_const].set(-0.5)
  
  phi=weightrbf@phi
  phi=phi.at[nid_const].set(-1.0)
  phi=jnp.clip(phi-0.1,-1.0,1.0)
  return phi

def init_phi_shell(coord,nid_const,weightrbf,n_msk=1):
  """
  connect : int (n,8)
  coord : float (n,3)
  """
  phi=jnp.ones(coord.shape[0]).at[nid_const].set(-1.0)
  for i in range(n_msk):
    phi=(weightrbf@phi).at[nid_const].set(-1.0)
  phi=jnp.clip(phi-0.1,-1.0,1.0)
  return phi

@jit
def _nid2dim_3d(nid):
  return (nid.repeat(3).reshape(-1,3)*3+jnp.arange(3)).flatten()

def get_mat_weight(coords_updated,connects_ls_raw,coords_ls_raw,idx):
  tri_coord=coords_ls_raw[connects_ls_raw[idx]]
  b_a=tri_coord[:,1]-tri_coord[:,0]
  c_a=tri_coord[:,2]-tri_coord[:,0]
  v_a=coords_updated-tri_coord[:,0]
  coeff=np.linalg.pinv(np.array([b_a,c_a]).transpose(1,0,2))
  coeff=(v_a[:,:,None]*coeff).sum(axis=1)
  weight=np.concatenate([1.-coeff.sum(axis=1,keepdims=True),coeff],axis=1)
  idx1=np.arange(coords_updated.shape[0]).repeat(3) #(3m1,)
  idx2=connects_ls_raw[idx].flatten() #(3m1,)
  data=weight.flatten() #(3m1,)
  mat=sp.sparse.csr_array((data,(idx1,idx2)),shape=(coords_updated.shape[0],coords_ls_raw.shape[0]))
  mat=BCSR.from_scipy_sparse(mat)
  return mat

def remove_zero_area_triangles(connects):
  """
  connects: np.array, shape=(n_triangles, 3)
  """
  idx_valid=((np.roll(connects,1,axis=1)-connects)!=0).all(axis=1)
  return connects[idx_valid]

def connects_valid(connects,nid_valid):
  """
  connects : (n,3)
  """
  nnode=connects.max()+1
  idx_nondiag=connects[:,[0,1,1,2,2,0]].reshape(-1,2)
  idx_diag=np.arange(nnode).repeat(2).reshape(-1,2)
  idx=np.concatenate([idx_nondiag,idx_diag],axis=0)
  data=np.ones(idx.shape[0],bool)
  adj=sp.sparse.csc_array((data,(idx[:,0],idx[:,1])),shape=(nnode,nnode))
  label=np.zeros(nnode,bool)
  label[nid_valid]=True
  while True:
    label_new=adj@label
    if (label_new==label).all():
      break
    label=label_new
  cid_valid=np.where(label[connects].all(axis=1))[0]
  return cid_valid

def check_vol(connect,coord,thresh=1e-10):
  vs=coord[connect]
  vol=np.linalg.det(vs[:,1:,:]-vs[:,0:1,:])
  if np.any(np.abs(vol)<thresh):
    raise ValueError('Zero volume element detected')

def get_lr(grd,loss,nid_var,ratio=0.2):
  lr=loss/(grd[nid_var]**2).sum()*ratio
  return lr

def get_lr_mod(grd,loss,nid_var,phi,ratio=0.2):
  grd_valid=grd.at[(phi==-1.0)*(grd>0.0)].set(0.0)
  grd_valid=grd_valid.at[(phi==1.0)*(grd<0.0)].set(0.0)
  lr=loss/(grd_valid[nid_var]**2).sum()*ratio
  return lr

def cosine_similarity(v_ref,v_trg):
  """
  v_ref : (ndim,nmode_ref)
  v_trg : (ndim,nmode_trg)
  """
  dots=v_ref.T@v_trg # (nmode_ref,nmode_trg)
  norm_ref=jnp.linalg.norm(v_ref,axis=0) # (nmode_ref,)
  norm_trg=jnp.linalg.norm(v_trg,axis=0) # (nmode_trg,)
  cossim=dots/(norm_ref[:,None]*norm_trg[None,:]) # (nmode_ref,nmode_trg)
  idx=jnp.argmax(jnp.abs(cossim),axis=1) # (nmode_ref,)
  return idx,np.max(jnp.abs(cossim),axis=1)