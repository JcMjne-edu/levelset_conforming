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
from lsto.fem.tetra4_fem import global_mat_full
from lsto.tetgen_tools import make_poly,postprocess_tetgen
from lsto.mesh_tools.mapping_surfacenode_full import mapping_surfacenode_full
from lsto.mesh_tools.mesh_postprocess_np import eliminate_lowaspect_triangle
from lsto.mesh_tools.mesh_utility import points_in_holes
from lsto.nastran_tools import run_nastran_eig,write_base_nastran
from lsto.levelset2stl_mat_tetra import mat_phi2face_tetra
from lsto.levelset_redivision_med import redivision,redivision_connect_coord
from lsto.loss_func import loss_cossim
from lsto.stl_tools import stl_from_connect_and_coord
from lsto.mesh_tools.preprocess import mesh3d_to_coord_and_connect,reconstruct_coordinates,connect2adjoint,cs_rbf_adjoint
from lsto.mesh_tools import meshbuilder,mesh_diff,mapping_dist,mapping_ls,mesh_smoothing
import logging
import numpy as np
import scipy as sp

from jax import config
config.update("jax_enable_x64", True)
logging.basicConfig(filename='lsto.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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
    os.makedirs('./tetgen',exist_ok=True)
    os.makedirs('./nastran',exist_ok=True)

  def get_phi0(self,lx,ly,m,val_hole=10.0):
    phi=init_phi_uniform_xy(self.connects_ls_str,self.coords_ls_str,self.nid_const,self.weightrbf,lx,ly,m,val_hole)
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
    coords_index,connects_ls=mesh3d_to_coord_and_connect(stop_gradient(mesh3d),round_f=8)
    detect_hole(connects_ls)
    coords_ls=mesh3d.reshape(-1,3)[coords_index]
    self.coords_ls_raw=stop_gradient(coords_ls); self.connects_ls_raw=connects_ls

    connects_ls_raw=remove_zero_area_triangles(connects_ls)
    coords_updated,connects_updated=mesh_smoothing.mesh_smoothing(stop_gradient(coords_ls),connects_ls_raw,self.target_length)
    u,inv=np.unique(connects_updated,return_inverse=True)
    connects_updated=inv.reshape(connects_updated.shape)
    #self.coords_updated=coords_updated; self.connects_updated=connects_updated
    self.mapping1_input=(coords_updated,np.asarray(connects_ls_raw),np.asarray(stop_gradient(coords_ls)))
    #idx=mapping_ls.tri_idx(coords_updated,connects_ls_raw,stop_gradient(coords_ls))
    #mat_weight_ls=get_mat_weight(coords_updated,connects_ls_raw,stop_gradient(coords_ls),idx)
    #coords_ls=mat_weight_ls@coords_ls

    #coords_ls,connects_ls=mesh_postprocess_jx(coords_ls,connects_ls,thresh_l=1e-2,thresh_v=1e-2)
    #self.coords_ls=stop_gradient(coords_ls)
    #self.connects_ls=connects_updated
    logging.info('Mesh diff start')
    coords_closed,connects_closed=mesh_diff.mesh_diff(self.v_geom,self.c_geom,coords_updated,connects_updated[:,::-1],True,self.target_length)
    logging.info('Mesh diff done')
    _,inv=np.unique(connects_closed,return_inverse=True)
    connects_closed=inv.reshape(connects_closed.shape)
    stl_from_connect_and_coord(connects_closed,coords_closed).save('./stl/check_raw.stl')
    connects_closed=eliminate_lowaspect_triangle(connects_closed,coords_closed,1e-2)
    logging.info('Mesh refinement done')
    #self.connects_closed=connects_closed
    self.coords_updated=coords_updated; self.connects_updated=connects_updated
    stl_from_connect_and_coord(connects_closed,coords_closed).save('./stl/check.stl')
    
    self.coords_hole=points_in_holes(coords_updated,connects_updated)
    return coords_closed,connects_closed,coords_ls
  
  def preprocess(self,phi,eigenanalysis=True):
    #self.phi=phi
    #_phi=self.weightrbf@phi
    #_phi=_phi.at[self.nid_const].set(jnp.clip(_phi[self.nid_const],a_max=-0.1))
    #_phi_ex=redivision(_phi,self.connects_ls_str)
    #numerator,denominator,offset=mat_phi2face_tetra(np.asarray(stop_gradient(_phi_ex)),self.connect_ls_ex)
    #mesh3d=((numerator@_phi_ex)@self.vertices_ex[offset])/(denominator@_phi_ex)[:,:,None] 
    #coords_index,connects_ls=mesh3d_to_coord_and_connect(stop_gradient(mesh3d))
    #coords_ls=mesh3d.reshape(-1,3)[coords_index]
    #self.coords_ls_raw=stop_gradient(coords_ls); self.connects_ls_raw=connects_ls
    #coords_ls,connects_ls=mesh_postprocess_jx(coords_ls,connects_ls,thresh_l=1.2e-1,thresh_v=1e-1)
    #self.coords_ls=stop_gradient(coords_ls)
    #self.connects_ls=connects_ls
    #logging.info('Mesh diff start')
    #coords_closed,connects_closed=mesh_diff.mesh_diff(self.v_geom,self.c_geom,np.array(stop_gradient(coords_ls)),np.array(stop_gradient(connects_ls))[:,::-1],True)
    #logging.info('Mesh diff done')
    #_,inv=np.unique(connects_closed,return_inverse=True)
    #connects_closed=inv.reshape(connects_closed.shape)
    #self.coords_closed=coords_closed
    #self.connects_closed_raw=connects_closed
    #mesh=stl_from_connect_and_coord(self.connects_closed_raw,self.coords_closed)
    #mesh.save('./check_raw.stl')
    #connects_closed=eliminate_lowaspect_triangle(connects_closed,coords_closed,1e-2)
    #logging.info('Mesh refinement done')
    #self.connects_closed=connects_closed
    #mesh=stl_from_connect_and_coord(self.connects_closed,self.coords_closed)
    #mesh.save('./check.stl')

    coords_closed,connects_closed,coords_ls=self.load_phi(phi)
    #self.coords_ls=stop_gradient(coords_ls)
    #self.connects_ls=connects_ls
    
    #self.coords_closed=coords_closed
    #nid_valid=np.where(coords_closed[:,1]==0.0)[0]
    #cid_valid=connects_valid(connects_closed,nid_valid)
    #print(connects_closed.shape)
    #print(cid_valid.shape)
    #nodes_tet,elems_tet=_tetgen_run(coords_closed,connects_closed[cid_valid],self.coords_hole)
    nodes_tet,elems_tet,faces_tet=_tetgen_run(coords_closed,connects_closed,self.coords_hole)
    self.nodes_tet=nodes_tet
    self.elems_tet=elems_tet
    self.faces_tet=faces_tet
    if eigenanalysis:
      self.eig_thread=CustomThread(target=run_nastran_eig,args=(elems_tet,nodes_tet,self.nastran_exe_path))
      self.eig_thread.start()
    check_vol(elems_tet,nodes_tet)
    print('Tetgen done')
    mat_weight,nid_valid_tet,self.nid_surf_tet=mapping_surfacenode_full(self.connects_updated,self.coords_updated,coords_closed,elems_tet,nodes_tet,faces_tet)
    
    self.set_target()
    self.dim_spc=_nid2dim_3d(jnp.where(nodes_tet[:,1]==0.0)[0])
    idx=mapping_ls.tri_idx(*self.mapping1_input)
    mat_weight_ls=get_mat_weight(*self.mapping1_input,idx)
    coords_ls=mat_weight_ls@coords_ls
    coord_fem=reconstruct_coordinates(coords_ls,jnp.array(self.nodes_tet),nid_valid_tet,mat_weight)
    coord_fem=coord_fem+0.0*phi.sum()
    matKg,matMg,mass=global_mat_full(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    return matKg,matMg,mass
  
  def main_eigval(self,phi):
    matKg,matMg,_=self.preprocess(phi)
    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matKg.data,matKg.indices,matMg,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.sol_eigvecs=sol_eigvecs
    self.sol_eigvals=sol_eigvals
    loss_eigval=(jnp.abs(v/self.v_trg[:self.num_mode_trg]-1.)).mean()/0.03
    self.v=v; self.w=w
    print(f'loss : {stop_gradient(loss_eigval):.3f} {np.round(stop_gradient(v),3)}')
    return loss_eigval
  
  def main_eigvec(self,phi):
    matKg,matMg,_=self.preprocess(phi)
    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matKg.data,matKg.indices,matMg,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.sol_eigvecs=sol_eigvecs; self.sol_eigvals=sol_eigvals
    self.v=v; self.w=w
    self.matKg=matKg; self.matMg=matMg
    loss_eigvec=loss_cossim(w[self.dim_active_full],self.w_trg[:,:self.num_mode_trg])/0.0001
    print(f'loss : {stop_gradient(loss_eigvec):.3f} {np.round(stop_gradient(v),3)}')
    return loss_eigvec
  
  def main_mass(self,phi):
    _,_,mass=self.preprocess(phi,eigenanalysis=False)
    loss_mass=jnp.abs(mass/self.mass_trg-1.)/0.03
    print(f'loss : {stop_gradient(loss_mass):.3f} {stop_gradient(mass)/self.mass_scale}/{self.mass_trg/self.mass_scale}')
    return loss_mass
  
  def main_static(self,phi):
    matKg,_,_=self.preprocess(phi,eigenanalysis=False)
    self.matKg=matKg
    kg_reduced=reduction_K(matKg,self.dim_active_rom,self.dim_spc)
    compliance_reduced=jnp.linalg.inv(kg_reduced)
    loss_static=((compliance_reduced-self.comp_trg)**2).sum()
    loss_static_scale=stop_gradient(compliance_reduced[self.dim_ref_comp,self.dim_ref_comp])/self.ref_comp
    loss_static_scale=(1.-loss_static_scale)
    self.compliance_reduced=compliance_reduced
    print('loss : ',stop_gradient(loss_static),stop_gradient(loss_static_scale))
    return loss_static
  
  def set_target_raw(self,k_l,k_p,coord_trg,nid_trg_rom,nid_trg_eig,comp_trg,mass,v,w):
    self.k_t,self.k_m,k_f=aeroelastic_scaling_wt(k_l,k_p)
    self.coord_trg_rom=coord_trg[nid_trg_rom]
    self.coord_trg_eig=coord_trg[nid_trg_eig]
    self.mass_trg=mass*self.k_m*self.mass_scale
    self.v_trg=v**2/self.k_t**2/self.mass_scale
    self.comp_trg=comp_trg*self.k_t**2/self.k_m
    self.dim_ref_comp=jnp.argmax(jnp.diag(self.comp_trg))
    self.ref_comp=self.comp_trg[self.dim_ref_comp,self.dim_ref_comp]
    k=w.shape[0]
    self.w_trg=w[:,nid_trg_eig].reshape(k,-1).T #(n_v*3,k)
    
  def set_target(self):
    nid_out_local_all_rom,_=mapping_dist.calc_dist(self.coord_trg_rom,self.nodes_tet[self.nid_surf_tet])
    self.surface_nid_identical_rom=self.nid_surf_tet[nid_out_local_all_rom]
    self.dim_active_rom=_nid2dim_3d(self.surface_nid_identical_rom)
    nid_local_eig,_=mapping_dist.calc_dist(self.coord_trg_eig,self.nodes_tet[self.nid_surf_tet])
    self.nid_tet_eig=self.nid_surf_tet[nid_local_eig]
    self.dim_active_full=_nid2dim_3d(self.nid_tet_eig)

def _tetgen_run(coords,connects,hole=None):
  make_poly('./tetgen/temp.poly',coords,connects,hole)
  #subprocess.run(['tetgen', '-q10.0F', 'tetgen/temp.poly'],stdout=subprocess.PIPE)
  out=os.system('tetgen -q5.0a0.2 ./tetgen/temp.poly > tetgen_output.log 2>&1')
  if out!=0:
    logging.error(f'Tetgen failed with code {out}')
    raise ValueError('Tetgen failed')
  nodes_tet,elems_tet,faces_tet=postprocess_tetgen('./tetgen','temp',1)
  shutil.move('./tetgen/temp.poly','./tetgen/temp_OOD.poly')
  shutil.move('./tetgen/temp.1.node','./tetgen/temp_OOD.1.node')
  shutil.move('./tetgen/temp.1.ele','./tetgen/temp_OOD.1.ele')
  return nodes_tet,elems_tet,faces_tet

def init_phi_uniform_xy(connect,coord,nid_const,weightrbf,lx,ly,m,val_hole=10.0):
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
  
  phi=weightrbf@phi
  phi=phi.at[nid_const].set(-1.0)
  phi=jnp.clip(phi,-1.0,1.0)
  return phi

def init_phi_uniform_xyz(connect,coord,nid_const,weightrbf,lx,ly,lz,m,val_hole=10.0):
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
