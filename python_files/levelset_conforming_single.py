from jax import custom_vjp
from jax.lax import stop_gradient
import os
from python_files.aeroelastic_scaling import *
from python_files.custom_eig_external_general import custom_eigsh_external
from python_files.custom_thread import *
from python_files.fem_tools import reduction_K
from python_files.tetra4_fem import global_mat_full
from python_files.tetgen_tools import *
from python_files.mapping_surfacenode import *
from python_files.mesh_postprocess_jax import mesh_postprocess_jx
from python_files.mesh_utility import *
from python_files.nastran_tools import run_nastran_eig
from python_files.preprocess_mesh import *
from python_files.levelset2stl_mat_tetra import mat_phi2face_tetra
from python_files.levelset_redivision import redivision,redivision_connect_coord
from python_files.loss_func import loss_cossim
import shutil
import subprocess
import triangle as tr

from jax import config
config.update("jax_enable_x64", True)

class LSTP_conforming:
  def __init__(self,wing3d,trimesh,meshbuilder,dmul):
    self.wing3d=wing3d
    self.trimesh=trimesh
    self.coord_ls_str=meshbuilder.coordinates
    self.connect_ls_str=meshbuilder.connectivity
    self.connect_ls_ex,self.coord_ls_ex=redivision_connect_coord(self.connect_ls_str,self.coord_ls_str)
    adjoint=connect2adjoint(self.connect_ls_str)
    self.weightrbf=cs_rbf_adjoint(self.coord_ls_str/meshbuilder.length_lattice,dmul,adjoint) #(nv,nv)
    self.nid_const=meshbuilder.nid_const
    _idx_tetra=np.array([0,4,5,7,0,1,3,7,0,5,1,7,1,2,3,6,1,6,3,7,1,5,6,7])
    connect_ls_tetra=self.connect_ls_str[:,_idx_tetra].reshape(-1,4) #(ne,4)
    self.vertices=self.coord_ls_str[connect_ls_tetra] #(ne,4,3)
    self.vertices_ex=self.coord_ls_ex[self.connect_ls_ex] #(ne,4,3)
    self.connect_ls_tetra=np.asarray(connect_ls_tetra)
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
  
  def setup(self,phi):
    _phi=self.weightrbf@phi
    _phi=_phi.at[self.nid_const].set(jnp.clip(_phi[self.nid_const],a_max=-0.1))
    _phi_ex=redivision(_phi,self.connect_ls_str)
    numerator,denominator,offset=mat_phi2face_tetra(np.asarray(stop_gradient(_phi_ex)),self.connect_ls_ex)
    mesh3d=((numerator@_phi_ex)@self.vertices_ex[offset])/(denominator@_phi_ex)[:,:,None] 
    coords_index,connects_ls=mesh3d_to_coord_and_connect(stop_gradient(mesh3d))
    coords_ls=mesh3d.reshape(-1,3)[coords_index]
    coords_ls,connects_ls=mesh_postprocess_jx(coords_ls,connects_ls,thresh_l=1.2e-1,thresh_v=1e-1)
    self.coords_hole=points_in_holes(stop_gradient(coords_ls),connects_ls)
    self.coords_ls=stop_gradient(coords_ls)
    self.connects_ls=connects_ls
    return _phi_ex,coords_ls
  
  def preprocess(self,_phi,eigenanalysis=True):
    segment=extract_root_edge(self.connects_ls,self.coords_ls) 
    #coord_hole=(self.coord_ls_str[(self.coord_ls_str[:,1]==0.0)*(_phi>0.0)])[:,[0,2]]
    coord_hole=(self.coord_ls_ex[(self.coord_ls_ex[:,1]==0.0)*(_phi>0.0)])[:,[0,2]]
    coord_outer2d=self.trimesh.coord[self.wing3d.root_edge_nid][:,[0,2]]
    coord_merged=np.concatenate([coord_outer2d,self.coords_ls[:,[0,2]]])
    seg1=np.array([np.arange(coord_outer2d.shape[0]-1),np.arange(1,coord_outer2d.shape[0])]).T
    seg1=np.concatenate([seg1,np.array([coord_outer2d.shape[0]-1,0]).reshape(-1,2)])
    seg=np.concatenate([seg1,segment+coord_outer2d.shape[0]])
    nid_used,u_inv=np.unique(seg,return_inverse=True)
    coord_merged=coord_merged[nid_used] #(n_v,2)
    seg=u_inv.reshape(-1,2)
    coord_hole=coord_hole[isinside2d(coord_hole,coord_merged[seg])]
    if coord_hole.shape[0]!=0:
      tr_input=dict(vertices=np.array(coord_merged),segments=np.array(seg),holes=np.array(coord_hole))
    else:
      tr_input=dict(vertices=np.array(coord_merged),segments=np.array(seg))
    self.tr_input=tr_input
    triangulation=tr.triangulate(tr_input,'p')
    vertices=triangulation['vertices']
    connect_cap=triangulation['triangles']
    coords_cap=np.zeros((vertices.shape[0],3))
    coords_cap[:,[0,2]]=vertices
    coords_closed=np.concatenate([self.coords_ls,self.trimesh.coord,coords_cap])
    n1=self.coords_ls.shape[0]
    n2=n1+self.trimesh.coord.shape[0]
    connects_closed=np.concatenate([self.connects_ls,self.wing3d.connectivity_surface_open+n1,connect_cap+n2])
    connects_closed,coords_closed=sweep_node(connects_closed,coords_closed)
    mesh_data=stl_from_connect_and_coord(connects_closed[:,::-1],coords_closed)
    self.coords_closed=coords_closed
    self.connects_closed=connects_closed
    mesh_data.save('./temp.stl')
    nodes_tet,elems_tet=_tetgen_run(coords_closed,connects_closed,np.asarray(self.coords_hole))
    elems_tet,nodes_tet=_eliminate_unused_node(elems_tet,nodes_tet)
    if eigenanalysis:
      self.eig_thread=CustomThread(target=run_nastran_eig,args=('./nastran',elems_tet,nodes_tet,self.young,self.poisson,self.rho,self.num_mode_ref))
      self.eig_thread.start()
    self.nodes_tet=jnp.asarray(nodes_tet)
    self.elems_tet=jnp.asarray(elems_tet)
    nid_identical_inside=nid_identical(self.coords_ls,nodes_tet)
    surface_nid_in,surface_nid_out=nid_in_and_out(self.nodes_tet,self.elems_tet,nid_identical_inside)
    self.surface_nid_out=surface_nid_out
    self.surface_nid_in=surface_nid_in
    self.set_target()
    self.nid_identical_inside=nid_identical_inside
    surface_mapping_in,_=mapping_surfacenode_mod(self.coords_ls,self.connects_ls,nodes_tet,elems_tet,nid_identical_inside,surface_nid_in)
    self.dim_spc=_nid2dim_3d(jnp.where(nodes_tet[:,1]==0.0)[0])
    return surface_nid_in,surface_mapping_in
  
  def main_eigval(self,phi):
    _phi_ex,coords_ls=self.setup(phi)
    nid_variable,surface_mapping_in=self.preprocess(stop_gradient(_phi_ex))
    coord_fem=reconstruct_coordinates(coords_ls,self.nodes_tet,nid_variable,surface_mapping_in)
    matKg,matMg,_=global_mat_full(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matKg.data,matKg.indices,matMg,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.sol_eigvecs=sol_eigvecs
    self.sol_eigvals=sol_eigvals
    loss_eigval=(jnp.abs(v/self.v_trg[:self.num_mode_trg]-1.)).mean()/0.03
    self.v=v; self.w=w
    print(f'loss : {stop_gradient(loss_eigval):.3f} {np.round(stop_gradient(v),3)}')
    return loss_eigval
  
  def main_eigvec(self,phi):
    _phi_ex,coords_ls=self.setup(phi)
    nid_variable,surface_mapping_in=self.preprocess(stop_gradient(_phi_ex))
    coord_fem=reconstruct_coordinates(coords_ls,self.nodes_tet,nid_variable,surface_mapping_in)
    matKg,matMg,_=global_mat_full(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    sol_eigvecs,sol_eigvals=self.eig_thread.join()
    v,w=custom_eigsh_external(matKg.data,matKg.indices,matMg,sol_eigvecs,sol_eigvals,self.num_mode_trg)
    self.sol_eigvecs=sol_eigvecs; self.sol_eigvals=sol_eigvals
    self.v=v; self.w=w
    self.matKg=matKg; self.matMg=matMg
    loss_eigvec=loss_cossim(w[self.dim_active_full],self.w_trg[:,:self.num_mode_trg])/0.0001
    print(f'loss : {stop_gradient(loss_eigvec):.3f} {np.round(stop_gradient(v),3)}')
    return loss_eigvec
  
  def main_mass(self,phi):
    _phi_ex,coords_ls=self.setup(phi)
    nid_variable,surface_mapping_in=self.preprocess(stop_gradient(_phi_ex),eigenanalysis=False)
    self.nid_variable=nid_variable
    self.surface_mapping_in=surface_mapping_in
    self.coords_ls=coords_ls
    coord_fem=reconstruct_coordinates(coords_ls,self.nodes_tet,nid_variable,surface_mapping_in)
    _,_,mass=global_mat_full(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    loss_mass=jnp.abs(mass/self.mass_trg-1.)/0.03
    print(f'loss : {stop_gradient(loss_mass):.3f} {stop_gradient(mass)/self.mass_scale}/{self.mass_trg/self.mass_scale}')
    return loss_mass
  
  def main_static(self,phi):
    _phi_ex,coords_ls=self.setup(phi)
    nid_variable,surface_mapping_in=self.preprocess(stop_gradient(_phi_ex),eigenanalysis=False)
    self.dim_spc=_nid2dim_3d(jnp.where(self.nodes_tet[:,1]==0.0)[0])
    coord_fem=reconstruct_coordinates(coords_ls,self.nodes_tet,nid_variable,surface_mapping_in)
    matKg,_,_=global_mat_full(self.elems_tet,coord_fem,self.young,self.poisson,self.rho)
    kg_reduced=reduction_K(matKg,self.dim_active_rom,self.dim_spc)
    compliance_reduced=jnp.linalg.inv(kg_reduced)
    loss_static=((compliance_reduced-self.comp_trg)**2).sum()
    loss_static_scale=stop_gradient(compliance_reduced[self.dim_ref_comp,self.dim_ref_comp])/self.ref_comp
    loss_static_scale=(1.-loss_static_scale)
    self.compliance_reduced=compliance_reduced
    print('loss : ',stop_gradient(loss_static),stop_gradient(loss_static_scale))
    return loss_static
  
  def set_target_raw(self,k_l,k_p,coord_trg,nid_surf_trg_rom,nid_surf_trg_full,comp_trg,mass,v,w):
    self.k_t,self.k_m,k_f=aeroelastic_scaling_wt(k_l,k_p)
    self.coord_trg_surf_rom=coord_trg[nid_surf_trg_rom]
    self.coord_trg_surf_full=coord_trg[nid_surf_trg_full]
    self.mass_trg=mass*self.k_m*self.mass_scale
    self.v_trg=v**2/self.k_t**2/self.mass_scale
    self.comp_trg=comp_trg*self.k_t**2/self.k_m
    self.dim_ref_comp=jnp.argmax(jnp.diag(self.comp_trg))
    self.ref_comp=self.comp_trg[self.dim_ref_comp,self.dim_ref_comp]
    k=w.shape[0]
    self.w_trg=w[:,nid_surf_trg_full].reshape(k,-1).T #(n_v*3,k)
    
  def set_target(self):
    nid_out_local_all_rom=nid_identical(self.coord_trg_surf_rom,self.nodes_tet[self.surface_nid_out])
    self.surface_nid_identical_rom=self.surface_nid_out[nid_out_local_all_rom]
    self.dim_active_rom=_nid2dim_3d(self.surface_nid_identical_rom)
    nid_out_local_all_full=nid_identical(self.coord_trg_surf_full,self.nodes_tet[self.surface_nid_out])
    self.surface_nid_identical_full=self.surface_nid_out[nid_out_local_all_full]
    self.dim_active_full=_nid2dim_3d(self.surface_nid_identical_full)
    
def _tetgen_run(coords,connects,hole=None):
  make_poly('./tetgen/temp.poly',coords,connects,hole)
  subprocess.run(['tetgen', '-q5.0', 'tetgen/temp.poly'],stdout=subprocess.PIPE)
  nodes_tet,elems_tet=postprocess_tetgen('./tetgen','temp',1)
  shutil.move('./tetgen/temp.poly','./tetgen/temp_OOD.poly')
  shutil.move('./tetgen/temp.1.node','./tetgen/temp_OOD.1.node')
  shutil.move('./tetgen/temp.1.ele','./tetgen/temp_OOD.1.ele')
  return nodes_tet,elems_tet

def nid_in_and_out(coord,connect,nid_identical_inside):
  """
  coord : (n,3)
  connect : (m,4)
  ref_coord_in : (n_in,3)
  """
  label=jnp.zeros(coord.shape[0],int) #(n,)
  label=label.at[nid_identical_inside].set(1)
  faces=connect[:,[0,1,3,1,2,3,0,2,3,0,2,1]].reshape(-1,3)
  faces_sorted=jnp.sort(faces,axis=1)
  root_nid=jnp.where(coord[:,1]==0.0)[0]
  unique_face,face_counts=jnp.unique(faces_sorted,axis=0,return_counts=True)
  root_fid=jnp.where(jnp.isin(unique_face,root_nid).all(axis=1))[0]
  surface_fid=jnp.where(face_counts==1)[0]
  surface_fid=jnp.setdiff1d(surface_fid,root_fid)
  surface_face=unique_face[surface_fid]
  edges=surface_face[:,[0,1,1,2,2,0]].reshape(-1,2)
  edges=jnp.unique(jnp.sort(edges,axis=1),axis=0)
  nid_non_root=(coord[:,1]!=0.0)
  msk_edge_valid=nid_non_root[edges].any(axis=1) # (n_edge,)
  edges=edges[msk_edge_valid]
  idx=jnp.concatenate((edges,edges[:,::-1]))
  data=jnp.ones(idx.shape[0],int)
  shape=(coord.shape[0],coord.shape[0])
  adj_mat=BCOO((data,idx),shape=shape)
  adj_mat=convertBCOO2BCSR(adj_mat)
  s=0
  while True:
    new_label=adj_mat@label
    label=new_label.at[new_label>1].set(1)
    if s==jnp.sum(label):
      break
    s=jnp.sum(label)
  nid_in=jnp.where(label==1)[0]
  nid_out=jnp.setdiff1d(jnp.unique(surface_face),nid_in)
  return nid_in,nid_out

@jit
def nid_identical(trg,ref,max_elem=2**32):
  """
  Calculate the node id of ref that is identical to trg

  trg : float (n,3)
  ref : float (m,3)
  """
  num_v_sep=max_elem//(ref.shape[0])
  num_iter=int(np.ceil(trg.shape[0]/num_v_sep))
  seps=np.arange(0,num_iter+1)*num_v_sep
  ids=jnp.zeros(trg.shape[0],int)
  _ref=jnp.asarray(ref)
  _trg=jnp.asarray(trg)
  for i in range(num_iter):
    tg=_trg[seps[i]:seps[i+1]]
    ids=ids.at[seps[i]:seps[i+1]].set(_nid_identical(tg,_ref))
  return ids

@jit
def _nid_identical(tg,rf):
  diff=jnp.abs(tg[:,None,:]-rf[None,:,:]).sum(axis=-1) # (n2,m)
  idx=jnp.argmin(diff,axis=1) # (n2,)
  return idx

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

@jit
def _nid2dim_3d(nid):
  return (nid.repeat(3).reshape(-1,3)*3+jnp.arange(3)).flatten()

def _eliminate_unused_node(connect,coord):
  unique_idx,new_connect=np.unique(connect,return_inverse=True,)
  new_connect=new_connect.reshape(-1,connect.shape[1])
  new_coord=coord[unique_idx]
  return new_connect,new_coord

def isinside2d(v,seg_v):
  """
  v : (n,2)
  seg_v : (m,2,2)
  """
  seg_not_vertical=np.where(seg_v[:,0,0]!=seg_v[:,1,0])[0]
  seg_v=seg_v[seg_not_vertical,None] # (m,1,2,2)
  msk_is_between=(v[:,0]>=seg_v[:,:,:,0].min(axis=-1))*(v[:,0]<seg_v[:,:,:,0].max(axis=-1)) # (m,n)
  intercept_y=seg_v[:,:,0,1]+(v[:,0]-seg_v[:,:,0,0])*(seg_v[:,:,1,1]-seg_v[:,:,0,1])/(seg_v[:,:,1,0]-seg_v[:,:,0,0]) # (m,n)
  msk_is_upper=(v[:,1]>intercept_y) # (m,n)
  counts=(msk_is_upper*msk_is_between).sum(axis=0) # (n,)
  return counts%2==0

def sweep_node(connect,coord):
  """
  connect : (n,3)
  coord : (m,3)
  """
  u_coord,u_inv=np.unique(coord,axis=0,return_inverse=True)
  u_connect=u_inv.flatten()[connect]
  return u_connect,u_coord

@custom_vjp
def grad_check(data):
  return data

def grad_check_fwd(data):
  return data,None

def grad_check_bwd(res,g):
  print(jnp.abs(g).max())
  return (g,)

grad_check.defvjp(grad_check_fwd,grad_check_bwd)
