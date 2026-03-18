import os, json
import numpy as np
from jax.lax import stop_gradient
import jax.numpy as jnp
import logging.config
from logging import getLogger
from lsto.aeroelastic_scaling import aeroelastic_scaling_wt
from lsto.custom_eig_external_general import custom_eigsh_external
from lsto.custom_thread import CustomThread
from lsto.fem.fem_tools import reduction_K
from lsto.fem.tetra4_fem import gmat_preprocess
from lsto.levelset2stl_mat_tetra import mat_phi2face_tetra
from lsto.levelset_redivision_med import redivision
from lsto.loss_func import loss_cossim
from lsto.mesh_tools import mapping_dist
from lsto.mesh_tools.improve_aspect_ratio import improve_aspect_fast
from lsto.mesh_tools.mapping_surfacenode_fast import mapping_surfacenode_fast
from lsto.mesh_tools.mesh_utility import elim_closed_surface
from lsto.mesh_tools.mesh_postprocess_jax import resolve_flattened_region, mesh_postprocess_jx
from lsto.mesh_tools.preprocess import extract_root_edge, mesh3d_to_coord_and_connect, reconstruct_coordinates
from lsto.nastran_tools import run_nastran_eig,write_base_nastran
from lsto.stl_tools import stl_from_mesh3d, stl_from_connect_and_coord
from lsto.tetgen_tools import eliminate_dup_node_elems, tetgen_run
import triangle as tr

with open('./src/lsto/log/log_config.json', 'r') as f:
  config = json.load(f)
  logging.config.dictConfig(config)

def detect_hole_open(connect,coord):
  """
  Check if the mesh has holes
  connect : (n,3) int
  coord : (m,3) float
  """
  msk_nid_root=(coord[:,1]<1e-2)
  edge=connect[:,[0,1,1,2,2,0]].reshape(-1,2)
  edge=np.sort(edge,axis=1)
  u,counts=np.unique(edge,axis=0,return_counts=True)
  msk_eid_open=msk_nid_root[u].all(axis=1)
  if (counts[~msk_eid_open]!=2).any():
    raise ValueError('Hole detected!')

class LSOptimizer:
  def __init__(self,connect_ls_str,nid_const,nid_var,connect_ls_ex,
               coord_ls_ex,v_geom,c_geom,weightrbf,target_length=0.2,angle_overhang=45.0,
               axis_overhang=jnp.array([0.,1.0,0.]),angle_edge=40.):
    """
    v_geom: (nvg,3) vertices on the geometry mesh
    c_geom: (neg,3) connectivity of the geometry mesh
    """
    self.connect_ls_str=connect_ls_str
    self.nid_const=nid_const
    self.nid_var=nid_var
    self.connect_ls_ex=connect_ls_ex
    self.weightrbf=weightrbf
    
    self.vertices_ex=coord_ls_ex[self.connect_ls_ex] #(ne,4,3)
    self.v_geom=v_geom
    self.c_geom=c_geom
    self.target_length=target_length
    self.logger_main=getLogger('lsto_main')
    self.logger_aux=getLogger('lsto_aux')
    self.epoch=None
    os.makedirs('./tetgen',exist_ok=True)
    os.makedirs('./nastran',exist_ok=True)
    msk_root_geom=(self.v_geom[:,1]==0.0)
    msk_elim_tri=msk_root_geom[self.c_geom].all(axis=1)
    self.c_geom_open=self.c_geom[~msk_elim_tri]
    
    self.angle_overhang=angle_overhang
    self.axis_overhang=axis_overhang
    self.angle_edge=angle_edge

  def set_config(self,young,poisson,rho,nastran_exe_path,num_mode_trg=6,num_mode_ref=6):
    self.young=young
    self.poisson=poisson
    self.rho=rho
    self.nastran_exe_path=nastran_exe_path
    self.num_mode_trg=num_mode_trg
    self.num_mode_ref=num_mode_ref
    write_base_nastran(num_mode_ref,young,poisson,self.rho)

  def preprocess_phi(self,phi):
    _phi=self.weightrbf@phi
    _phi=_phi.at[self.nid_const].set(jnp.clip(_phi[self.nid_const],max=-0.1,))
    _phi_ex=redivision(_phi,self.connect_ls_str)
    numerator,denominator,offset=mat_phi2face_tetra(_nparray(_phi_ex),self.connect_ls_ex)
    mesh3d=((numerator@_phi_ex)@self.vertices_ex[offset])/(denominator@_phi_ex)[:,:,None]
    self.logger_aux.info(f'mesh3d generated ({mesh3d.shape})')
    stl_from_mesh3d(_nparray(mesh3d)).save('./stl/mesh3d.stl')
    for rf in [9,8,7,6,5,4]:
      try:
        coord_index,connect_ls=mesh3d_to_coord_and_connect(_nparray(mesh3d),round_f=rf)
        coord_ls=mesh3d.reshape(-1,3)[coord_index]
        detect_hole_open(connect_ls,coord_ls)
        break
      except ValueError:
        if rf==4:
          raise ValueError('Hole detected!!')
        continue
    self.coord_mesh3d=coord_ls
    self.connect_mesh3d=connect_ls
    self.logger_aux.info(f'mesh3d converted ({coord_ls.shape[0]})')
    connect_ls_closed=close_mesh(connect_ls,_nparray(coord_ls),None,True)
    self.logger_aux.info('mesh closed')
    # Mesh quality improvement
    self.connect_ls_closed=connect_ls_closed
    self.coord_ls_closed=coord_ls
    connect_ls_closed=improve_aspect_fast(connect_ls_closed,_nparray(coord_ls))
    stl_from_connect_and_coord(_nparray(connect_ls_closed),_nparray(coord_ls)).save('./stl/improve_aspect.stl')
    self.logger_aux.info('improve_aspect_fast done')
    coord_ls,connect_ls=mesh_postprocess_jx(connect_ls_closed,coord_ls,thresh_l=self.target_length)
    self.logger_aux.info('mesh_postprocess_jx done')
    connect_ls,coord_ls=resolve_flattened_region(connect_ls,coord_ls)
    connect_ls,coord_ls=elim_closed_surface(connect_ls,coord_ls)
    connect_ls,coord_ls=elim_flat_tet(connect_ls,coord_ls)
    self.loss_edge=penalty_angle(connect_ls,coord_ls,self.angle_edge)
    stl_from_connect_and_coord(_nparray(connect_ls),_nparray(coord_ls)).save('./stl/check.stl')
    coord_hole_2d_full=_nparray(get_coord_hole_2d(connect_ls,coord_ls))
    msk_face_root=get_msk_face_root(_nparray(connect_ls),_nparray(coord_ls))
    if msk_face_root.any():
      self.logger_aux.info('open boundary')
      connect_ls=connect_ls[~msk_face_root]
      connect_merged=jnp.vstack([connect_ls,self.c_geom_open+coord_ls.shape[0]])
      coord_merged=jnp.vstack([coord_ls,self.v_geom])
      self.nid_surf_geom=np.unique(self.c_geom_open)+coord_ls.shape[0]
      connect_merged=close_mesh(_nparray(connect_merged),_nparray(coord_merged),coord_hole_2d_full)
    else:
      connect_merged=jnp.vstack([connect_ls,self.c_geom+coord_ls.shape[0]])
      self.nid_surf_geom=np.unique(self.c_geom)+coord_ls.shape[0]
      coord_merged=jnp.vstack([coord_ls,self.v_geom])
    connect_marker=np.ones(connect_merged.shape[0])
    connect_marker[connect_ls.shape[0]:]=2
    self.connect_marker=connect_marker
    return coord_merged,connect_merged,coord_ls,connect_ls
  
  def preprocess(self,phi,eigenanalysis=True):
    coord_closed,connect_closed,coord_ls,connect_ls=self.preprocess_phi(phi)
    stl_from_connect_and_coord(_nparray(connect_closed),_nparray(coord_closed)).save('./stl/check_closed.stl')
    self.logger_aux.info('load phi finished')
    node_tet,elem_tet,face_tet=tetgen_run(_nparray(coord_closed),connect_closed,None,self.connect_marker)
    elem_tet=eliminate_dup_node_elems(elem_tet)
    self.node_tet=node_tet
    self.elem_tet=elem_tet
    self.face_tet=face_tet
    if eigenanalysis:
      self.eig_thread=CustomThread(target=run_nastran_eig,args=(elem_tet,node_tet,self.nastran_exe_path))
      self.eig_thread.start()
    self.logger_aux.info('tetgen finished')
    mat_weight,self.nid_surf_tet=mapping_surfacenode_fast(stop_gradient(connect_ls),stop_gradient(coord_ls),node_tet,face_tet)
    self.logger_aux.info('mapping_surfacenode_fast finished')
    self.set_target()
    self.logger_aux.info('set_target finished')
    self.dim_spc=nid2dim_3d(jnp.where(node_tet[:,1]==0.0)[0])
    coord_fem=reconstruct_coordinates(coord_ls,jnp.array(self.node_tet),self.nid_surf_tet,mat_weight)
    self.logger_aux.info('reconstruct_coordinates finished')
    matKg,matMg=gmat_preprocess(self.elem_tet,coord_fem,self.young,self.poisson,self.rho)
    self.logger_aux.info('gmat_preprocess finished')
    self.matKg=matKg; self.matMg=matMg
    self.loss_overhang=loss_overhang(connect_ls,coord_ls,self.angle_overhang,self.axis_overhang)
    return matKg,matMg
  
  def main(self,phi,include_static=True,include_eigval=True,include_eigvec=True):
    loss_total=self.loss_overhang
    msg_main =f'\nEpoch : {self.epoch}'
    matK,matM=self.preprocess(phi,eigenanalysis=include_eigval or include_eigvec)
    self.logger_aux.info('preprocess finished')
    if include_static:
      kg_reduced=reduction_K(matK,self.dim_active_rom,self.dim_spc)
      compliance_reduced=jnp.linalg.inv(kg_reduced)
      loss_static=(1.-(compliance_reduced*self.comp_trg).sum()/jnp.linalg.norm(compliance_reduced)/self.comp_trg_norm)*1e2
      msg_main+=f'\nloss (static) : {stop_gradient(loss_static):.3f} ({stop_gradient(jnp.linalg.norm(compliance_reduced)/self.comp_trg_norm):.5f})'
      loss_total+=loss_static

    if include_eigval or include_eigvec:
      sol_eigvecs,sol_eigvals=self.eig_thread.join()
      v,w=custom_eigsh_external(matK.data,matK.indices,matM,sol_eigvecs,sol_eigvals,self.num_mode_trg)
      self.logger_aux.info('Nastran calculation finished')
      cossim,aux=cosine_similarity(sol_eigvecs[self.dim_active_eig],self.w_trg)
      v_temp=v[cossim]
      w_temp=w[:,cossim]
      weight=1/jnp.arange(1,self.num_mode_trg)**0.4
      weight=weight/weight.sum()

      if include_eigval:
        loss_eigval=((v_temp/v_temp[0]/self.v_trg[:self.num_mode_trg]*self.v_trg[0]-1.)[1:]**2)@weight*1e4
        loss_total+=loss_eigval
        msg_main+=f'\nloss (eigval) : {stop_gradient(loss_eigval):.3f} ({stop_gradient(v[0]):.3f})'

      if include_eigvec:
        loss_eigvec=loss_cossim(w_temp[self.dim_active_eig],self.w_trg)/0.00001
        msg_main+=f'\nloss (eigvec) : {stop_gradient(loss_eigvec):.3f}'
        loss_total+=loss_eigvec

      fmt='{:6.2f}  '*(self.num_mode_trg)
      msg_main+=('\nratio: '+fmt.format(*stop_gradient(v_temp/v_temp[0])))
      msg_main+=('\n ref : '+fmt.format(*(self.v_trg[:self.num_mode_trg]/self.v_trg[0])))
      fmt='{:.4f}  '*aux.shape[0]
      msg_main+=('\n cos : '+fmt.format(*aux)+'\n')

    msg_main+=f'\nloss (overhang) : {stop_gradient(self.loss_overhang):.3f}'
    self.logger_main.info(msg_main)

    return loss_total
  
  def set_target_raw(self,k_l,k_p,coord_trg,nid_trg_rom,nid_trg_eig,comp_trg,mass,v,w):
    self.k_t,self.k_m,_=aeroelastic_scaling_wt(k_l,k_p)
    self.coord_trg_rom=coord_trg[nid_trg_rom]
    self.coord_trg_eig=coord_trg[nid_trg_eig]
    self.mass_trg=mass*self.k_m
    self.v_trg_raw=v
    self.v_trg=v/self.k_t**2
    self.comp_trg=comp_trg#*self.k_t**2/self.k_m
    self.comp_trg_norm=jnp.linalg.norm(self.comp_trg)
    self.dim_ref_comp=jnp.argmax(jnp.diag(self.comp_trg))
    self.ref_comp=self.comp_trg[self.dim_ref_comp,self.dim_ref_comp]
    k=w.shape[0]
    self.w_trg=w[:,nid_trg_eig].reshape(k,-1).T[:,:self.num_mode_trg] #(n_v*3,k)

  def set_target(self):
    nid_out_local_all_rom,_=mapping_dist.calc_dist(self.coord_trg_rom,self.node_tet[self.nid_surf_geom])
    self.surface_nid_identical_rom=self.nid_surf_geom[nid_out_local_all_rom]
    self.dim_active_rom=nid2dim_3d(self.surface_nid_identical_rom)
    nid_local_eig,_=mapping_dist.calc_dist(self.coord_trg_eig,self.node_tet[self.nid_surf_geom])
    self.nid_tet_eig=self.nid_surf_geom[nid_local_eig]
    self.dim_active_eig=nid2dim_3d(self.nid_tet_eig)
  
def close_mesh(connect,coord,coord_hole_2d=None,reverse=False):
  seg=extract_root_edge(connect,coord)
  if seg.shape[0]==0:
    return connect
  unid,inv=np.unique(seg,return_inverse=True)
  _coord=coord[unid]
  _seg=inv.reshape(-1,2)
  if coord_hole_2d is not None:
    tr_input=dict(vertices=_coord[:,[0,2]],segments=_seg,holes=coord_hole_2d)
  else:
    tr_input=dict(vertices=_coord[:,[0,2]],segments=_seg)
  triangulation=tr.triangulate(tr_input,'p')
  connect_cap=unid[triangulation['triangles']]
  if reverse:
    connect_cap=connect_cap[:,::-1]
  connect_closed=np.vstack((connect,connect_cap))
  return connect_closed

def _nparray(arr):
  return np.array(stop_gradient(arr),)

def get_msk_face_root(connect,coord):
  """
  connect : (n,3)
  coord : (m,3)
  """
  msk_nid_root=(coord[:,1]<1e-3) # (m,)
  msk_fid_root=(msk_nid_root[connect]).all(axis=1) # (n,)
  return msk_fid_root

def elim_flat_tet(connect,coord):
  edge=connect[:,[0,1,1,2,2,0]].reshape(-1,2)
  edge=jnp.sort(edge,axis=1)
  edge,inv,counts=jnp.unique(edge,axis=0,return_inverse=True,return_counts=True)
  msk_edge=(counts==1)
  msk_connect=msk_edge[inv].reshape(-1,3).any(axis=1)
  connect_valid=connect[~msk_connect]
  unid,inv=jnp.unique(connect_valid,return_inverse=True)
  coord=coord[unid]
  connect=inv.reshape(-1,3)
  return connect,coord

def loss_overhang(connect,coord,angle,axis):
  """
  Compute loss of overhang
  connect : (n,3) int
  coord : (m,3) float
  angle : float
  axis : (3,) float
  """
  v=coord[connect]
  v1=v[:,1]-v[:,0]
  v2=v[:,2]-v[:,0]
  normal=jnp.cross(v1,v2)
  area=jnp.linalg.norm(normal,axis=1)
  normal=normal/area[:,None] # (n,3)
  cosines=normal@axis # (n,)
  diff=cosines+np.cos(np.deg2rad(angle))
  msk=(diff<0.0)
  penalty=-(diff*area)[msk].sum()
  return penalty

def penalty_angle(connect,coord,angle):
  """
  Compute penalty of angle
  connect : (n,3) int
  coord : (m,3) float
  angle : float
  """
  edge=connect[:,[0,1,1,2,2,0]].reshape(-1,2) # (ne,2)
  edge=jnp.sort(edge,axis=1) # (ne,2)
  u_edge,inv,count=jnp.unique(edge,axis=0,return_inverse=True,return_counts=True)
  u_edge_aug=jnp.zeros((count.sum()//2,2),int)
  msk_eid=jnp.ones(count.sum()//2,bool)

  offset=0
  eid_dup=jnp.where(count!=2)[0]
  if eid_dup.shape[0]!=0:
    eid_dup=jnp.concatenate([np.array([0,]),eid_dup,np.array([count.shape[0]-1])])
    for i in range(eid_dup.shape[0]-1):
      u_edge_aug=u_edge_aug.at[offset+eid_dup[i]:offset+eid_dup[i+1]].set(u_edge[eid_dup[i]:eid_dup[i+1]])
      u_edge_aug=u_edge_aug.at[offset+eid_dup[i+1]:offset+eid_dup[i+1]+count[eid_dup[i+1]]//2].set(u_edge[eid_dup[i+1]])
      msk_eid=msk_eid.at[offset+eid_dup[i+1]:offset+eid_dup[i+1]+count[eid_dup[i+1]]//2].set(False)
      offset+=count[eid_dup[i+1]]//2
  else:
    u_edge_aug=u_edge
  edge_length=jnp.linalg.norm(coord[u_edge_aug[:,1]]-coord[u_edge_aug[:,0]],axis=1) # (ne,)
  map_e2f=jnp.arange(connect.shape[0]).repeat(3)[jnp.argsort(inv)].reshape(-1,2) #(ne,2)
  vs=coord[connect] # (n,3,3)
  norm=jnp.cross(vs[:,1]-vs[:,0],vs[:,2]-vs[:,0]) # (n,3)
  norm=norm/jnp.linalg.norm(norm,axis=1,keepdims=True) # (n,3)
  cosines=(norm[map_e2f[:,0]]*norm[map_e2f[:,1]]).sum(axis=1) # (ne,)
  theta=jnp.arccos(jnp.clip(cosines,min=-1.0+1e-6,max=1.0-1e-6)) # (ne,)
  msk=theta>np.deg2rad(180.-angle)
  penalty=((theta-np.deg2rad(angle))**2*edge_length)[msk*msk_eid].sum()
  return penalty

def get_coord_hole_2d(connect_ls,coord_ls,thresh=0.2):
  """
    connect_ls (m,3)
    coord_ls (n,3)
  """
  vertice=coord_ls[connect_ls] # (m,3,3)
  
  msk_vertice=(vertice[:,:,1]==0.0).all(axis=1) # (m',)

  vertice=vertice[msk_vertice] # (m',3,3)
  edge_length=jnp.linalg.norm(vertice-vertice[:,[1,2,0]],axis=2) # (m',3)
  max_edge_length=edge_length.max(axis=1) # (m',)
  area=jnp.linalg.norm(jnp.cross(vertice[:,1]-vertice[:,0],vertice[:,2]-vertice[:,0]),axis=1) # (m',)
  height=area/max_edge_length # (m',)
  vertice_valid=vertice[height>thresh] # (m'',3,3)
  coord_mid=vertice_valid.mean(axis=1) # (m'',3)
  return coord_mid[:,[0,2]] # (m'',2)

def nid2dim_3d(nid):
  return (nid.repeat(3).reshape(-1,3)*3+jnp.arange(3)).flatten()

def cosine_similarity(w_ref,w_trg):
  """
  w_ref : (ndim,nmode_ref)
  w_trg : (ndim,nmode_trg)
  """
  nmode_ref=w_ref.shape[1]
  nmode_trg=w_trg.shape[1]
  dots=w_ref.T@w_trg # (nmode_ref,nmode_trg)
  norm_ref=jnp.linalg.norm(w_ref,axis=0) # (nmode_ref,)
  norm_trg=jnp.linalg.norm(w_trg,axis=0) # (nmode_trg,)
  cossim=dots/(norm_ref[:,None]*norm_trg[None,:]) # (nmode_ref,nmode_trg)
  val=jnp.abs(cossim)
  arg=np.argsort(val,axis=1)[:,::-1]
  idx=[]
  for i in range(nmode_ref):
    for j in range(nmode_trg):
      if arg[i,j] not in idx:
        idx.append(arg[i,j])
        break
  return np.array(idx),val[np.arange(nmode_ref),idx]

def get_lr(grd,loss,nid_var,phi,ratio=0.2):
  grd_valid=grd.at[(phi==-1.0)*(grd>0.0)].set(0.0)
  grd_valid=grd_valid.at[(phi==1.0)*(grd<0.0)].set(0.0)
  lr=loss/(grd_valid[nid_var]**2).sum()*ratio
  return lr