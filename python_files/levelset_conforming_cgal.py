import os,sys
import shutil
import subprocess
#file_path=os.path.abspath(__file__)
#directory_path = os.path.dirname(file_path)
#cgal_path='/'.join(file_path.split('/')[:-2])+'/cgal'
#sys.path.append(directory_path)

from jax.lax import stop_gradient
from aeroelastic_scaling import *
from custom_eig_external_general import custom_eigsh_external
from custom_thread import CustomThread
from fem_tools import reduction_K
from tetra4_fem import global_mat_full
from tetgen_tools import *
from mapping_surfacenode_full import mapping_surfacenode_full
from mesh_postprocess_jax import mesh_postprocess_jx
from mesh_postprocess_np import merge_close_nodes
from mesh_utility import points_in_holes
from nastran_tools import run_nastran_eig
from preprocess_mesh import *
from levelset2stl_mat_tetra import mat_phi2face_tetra
from levelset_redivision import redivision,redivision_connect_coord
from loss_func import loss_cossim
import meshbuilder,mesh_diff,mapping_dist

from jax import config
config.update("jax_enable_x64", True)


class LSTP_conforming:
  def __init__(self,v_dspace,c_dspace,length_lattice,dmul,v_geom,c_geom):
    """
    v_dspace: (nv,3) vertices on the design space boundary
    c_dspace: (ne,4) connectivity of the design space
    v_geom: (nvg,3) vertices on the geometry mesh
    c_geom: (neg,3) connectivity of the geometry mesh
    """
    coords_ls_str,connects_ls_str,self.nid_const=meshbuilder.meshbuilder(v_dspace,c_dspace,*length_lattice)
    self.coord_ls_str=jnp.array(coords_ls_str.reshape(-1,3))
    self.connect_ls_str=jnp.array(connects_ls_str.reshape(-1,8))
    self.connect_ls_ex,self.coord_ls_ex=redivision_connect_coord(self.connect_ls_str,self.coord_ls_str)
    adjoint=connect2adjoint(self.connect_ls_str)
    self.weightrbf=cs_rbf_adjoint(self.coord_ls_str/length_lattice,dmul,adjoint) #(nv,nv)
    _idx_tetra=np.array([0,4,5,7,0,1,3,7,0,5,1,7,1,2,3,6,1,6,3,7,1,5,6,7])
    connect_ls_tetra=self.connect_ls_str[:,_idx_tetra].reshape(-1,4) #(ne,4)
    self.vertices=self.coord_ls_str[connect_ls_tetra] #(ne,4,3)
    self.vertices_ex=self.coord_ls_ex[self.connect_ls_ex] #(ne,4,3)
    self.connect_ls_tetra=np.asarray(connect_ls_tetra)
    self.v_geom=v_geom
    self.c_geom=c_geom
    os.makedirs('./tetgen',exist_ok=True)
    os.makedirs('./nastran',exist_ok=True)

  def set_material(self,young,poisson,rho,mass_scale=1.0):
    self.young=young
    self.poisson=poisson
    self.rho=rho*mass_scale
    self.mass_scale=mass_scale

  def set_num_modes(self,num_mode_trg,num_mode_ref):
    self.num_mode_trg=num_mode_trg
    self.num_mode_ref=num_mode_ref

  def load_phi(self,phi):
    _phi=self.weightrbf@phi
    _phi=_phi.at[self.nid_const].set(jnp.clip(_phi[self.nid_const],a_max=-0.1))

    numerator,denominator,offset=mat_phi2face_tetra(np.asarray(stop_gradient(_phi)),self.connect_ls_tetra)
    mesh3d=((numerator@_phi)@self.vertices[offset])/(denominator@_phi)[:,:,None] 
    coords_index,connects_ls=mesh3d_to_coord_and_connect(stop_gradient(mesh3d))
    coords_ls=mesh3d.reshape(-1,3)[coords_index]
    #coords_ls,connects_ls=mesh_postprocess_jx(coords_ls,connects_ls,thresh_l=1.2e-1,thresh_v=1e-1)
    return coords_ls,connects_ls
  
  def preprocess(self,phi,eigenanalysis=True):
    _phi=self.weightrbf@phi
    _phi=_phi.at[self.nid_const].set(jnp.clip(_phi[self.nid_const],a_max=-0.1))
    _phi_ex=redivision(_phi,self.connect_ls_str)
    numerator,denominator,offset=mat_phi2face_tetra(np.asarray(stop_gradient(_phi_ex)),self.connect_ls_ex)
    mesh3d=((numerator@_phi_ex)@self.vertices_ex[offset])/(denominator@_phi_ex)[:,:,None] 
    coords_index,connects_ls=mesh3d_to_coord_and_connect(stop_gradient(mesh3d))
    coords_ls=mesh3d.reshape(-1,3)[coords_index]
    coords_ls,connects_ls=mesh_postprocess_jx(coords_ls,connects_ls,thresh_l=1.2e-1,thresh_v=1e-1)
    coords_closed,connects_closed=mesh_diff.mesh_diff(self.v_geom,self.c_geom,np.array(stop_gradient(coords_ls)),np.array(stop_gradient(connects_ls))[:,::-1])
    connects_closed,coords_closed=merge_close_nodes(connects_closed,coords_closed,threashold=1e-6)
    self.coords_hole=points_in_holes(stop_gradient(coords_ls),connects_ls)
    self.coords_ls=stop_gradient(coords_ls)
    self.connects_ls=connects_ls
    
    self.coords_closed=coords_closed
    nodes_tet,elems_tet=_tetgen_run(coords_closed,connects_closed,self.coords_hole)
    if eigenanalysis:
      self.eig_thread=CustomThread(target=run_nastran_eig,args=('./nastran',elems_tet,nodes_tet,self.young,self.poisson,self.rho,self.num_mode_ref))
      self.eig_thread.start()
    self.nodes_tet=nodes_tet
    self.elems_tet=jnp.asarray(elems_tet)
    self.elems_tet=elems_tet
    mat_weight,nid_valid_tet,self.nid_surf_tet=mapping_surfacenode_full(np.array(self.connects_ls),np.array(self.coords_ls),coords_closed,elems_tet,nodes_tet)
    self.set_target()
    self.dim_spc=_nid2dim_3d(jnp.where(nodes_tet[:,1]==0.0)[0])
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
  subprocess.run(['tetgen', '-q5.0F', 'tetgen/temp.poly'],stdout=subprocess.PIPE)
  nodes_tet,elems_tet=postprocess_tetgen('./tetgen','temp',1)
  shutil.move('./tetgen/temp.poly','./tetgen/temp_OOD.poly')
  shutil.move('./tetgen/temp.1.node','./tetgen/temp_OOD.1.node')
  shutil.move('./tetgen/temp.1.ele','./tetgen/temp_OOD.1.ele')
  return nodes_tet,elems_tet

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
  phi=jnp.clip(phi,-1.0,1.0)
  return phi

@jit
def _nid2dim_3d(nid):
  return (nid.repeat(3).reshape(-1,3)*3+jnp.arange(3)).flatten()
