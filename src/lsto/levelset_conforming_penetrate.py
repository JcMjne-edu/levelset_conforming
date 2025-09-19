from lsto.levelset_conforming_cgal import *
from lsto.mesh_tools.improve_aspect_ratio import improve_aspect_fast
from lsto.mesh_tools.preprocess import extract_root_edge
import triangle as tr
from lsto.mesh_tools.mesh_utility import elim_closed_surface
from lsto.stl_tools import stl_from_mesh3d
from lsto.tetgen_tools import eliminate_dup_node_elems

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

class LSTP_conforming_penetrate(LSTP_conforming):
  def __init__(self,v_dspace,c_dspace,length_lattice,dmul,v_geom,c_geom,
    target_length=0.2,angle_overhang=45.0,axis_overhang=jnp.array([0.,1.0,0.]),angle_edge=40.):
    """
    v_dspace: (nv,3) vertices on the design space boundary
    c_dspace: (ne,4) connectivity of the design space
    v_geom: (nvg,3) vertices on the geometry mesh
    c_geom: (neg,3) connectivity of the geometry mesh
    """
    super().__init__(v_dspace,c_dspace,length_lattice,dmul,v_geom,c_geom,target_length)
    msk_root_geom=(self.v_geom[:,1]==0.0)
    msk_elim_tri=msk_root_geom[self.c_geom].all(axis=1)
    self.c_geom_open=self.c_geom[~msk_elim_tri]
    seg_outer=extract_root_edge(self.c_geom,self.v_geom)
    self.unid_outer,inv_outer=np.unique(seg_outer,return_inverse=True)
    self._coord_outer=self.v_geom[self.unid_outer]
    self._seg_outer=inv_outer.reshape(-1,2)
    self.msk_pid_root=(self.coord_ls_str[:,1]==0.0)

    nid_root=np.where(self.coord_ls_str[:,1]==0.0)[0]
    _,count=np.unique(self.connect_ls_str,return_counts=True)
    nid_var_additional=np.intersect1d(np.where(count==4)[0],nid_root)
    self.nid_const=np.setdiff1d(self.nid_const,nid_var_additional)
    self.nid_var=np.union1d(self.nid_var,nid_var_additional)
    self.angle_overhang=angle_overhang
    self.axis_overhang=axis_overhang
    self.angle_edge=angle_edge

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
    self.logger_aux.info(f'mesh3d converted ({coord_ls.shape[0]})')
    connect_ls_closed=close_mesh(connect_ls,_nparray(coord_ls),None,True)
    self.logger_aux.info('mesh closed')
    # Mesh quality improvement
    self.connect_ls_closed=connect_ls_closed
    self.coord_ls_closed=coord_ls
    connect_ls_closed=improve_aspect_fast(connect_ls_closed,_nparray(coord_ls))
    self.logger_aux.info('improve_aspect_fast done')
    coord_ls,connect_ls=mesh_postprocess_jx(connect_ls_closed,coord_ls,thresh_l=self.target_length,thresh_v=1e-1)
    self.logger_aux.info('mesh_postprocess_jx done')
    connect_ls,coord_ls=elim_closed_surface(connect_ls,coord_ls)
    connect_ls,coord_ls=elim_flat_tet(connect_ls,coord_ls)
    self.loss_edge=penalty_angle(connect_ls,coord_ls,self.angle_edge)
    stl_from_connect_and_coord(_nparray(connect_ls),_nparray(coord_ls)).save('./stl/check.stl')
    msk_face_root=get_msk_face_root(_nparray(connect_ls),_nparray(coord_ls))
    if msk_face_root.any():
      self.logger_aux.info('open boundary')
      coord_hole_2d_full=_nparray(coord_ls)[connect_ls[msk_face_root]].mean(axis=1)[:,[0,2]]
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

  def load_phi(self,phi):
    self.phi=phi
    _phi=self.weightrbf@phi
    _phi=_phi.at[self.nid_const].set(jnp.clip(_phi[self.nid_const],a_max=-0.1))
    _phi_ex=redivision(_phi,self.connect_ls_str)
    numerator,denominator,offset=mat_phi2face_tetra(np.asarray(stop_gradient(_phi_ex)),self.connect_ls_ex)
    mesh3d=((numerator@_phi_ex)@self.vertices_ex[offset])/(denominator@_phi_ex)[:,:,None] 
    self.mesh3d=mesh3d
    for rf in [10,9,8,7,6]:
      try:
        coord_index,connect_ls=mesh3d_to_coord_and_connect(stop_gradient(mesh3d),round_f=rf)
        detect_hole(connect_ls)
        break
      except ValueError:
        continue
    detect_hole(connect_ls)
    coord_ls=mesh3d.reshape(-1,3)[coord_index]
    self.coord_ls_raw=stop_gradient(coord_ls); self.connect_ls_raw=connect_ls
    stl_from_connect_and_coord(self.connect_ls_raw,self.coord_ls_raw).save('./stl/check_raw.stl')
    connect_ls=improve_aspect_fast(np.array(connect_ls),np.array(stop_gradient(coord_ls)))
    coord_ls,connect_ls=mesh_postprocess_jx(connect_ls,coord_ls,thresh_l=self.target_length,thresh_v=1e-1)
    self.coord_hole,connect_ls,coord_ls=points_in_holes(coord_ls,connect_ls)
    coord_closed=np.concatenate([stop_gradient(coord_ls),self.v_geom])
    connect_closed=np.concatenate([connect_ls,self.c_geom+coord_ls.shape[0]])
    connect_marker=np.ones(connect_closed.shape[0])
    connect_marker[connect_ls.shape[0]:]=2
    self.coord_closed=coord_closed; self.connect_closed=connect_closed; self.connect_marker=connect_marker
    self.coord_ls=stop_gradient(coord_ls); self.connect_ls=connect_ls
    stl_from_connect_and_coord(self.connect_ls,self.coord_ls).save('./stl/check.stl')
    self.nid_surf_geom=np.unique(self.c_geom)+self.coord_ls.shape[0]
    return coord_closed,connect_closed,coord_ls,connect_ls
  
  def preprocess(self,phi,eigenanalysis=True):
    #coord_closed,connect_closed,coord_ls,connect_ls=self.load_phi(phi)
    coord_closed,connect_closed,coord_ls,connect_ls=self.preprocess_phi(phi)
    self.logger_aux.info('load phi finished')
    #node_tet,elem_tet,face_tet=tetgen_run(_nparray(coord_closed),_nparray(connect_closed),self.coord_hole,self.connect_marker)
    node_tet,elem_tet,face_tet=tetgen_run(_nparray(coord_closed),connect_closed,None,self.connect_marker)
    elem_tet=eliminate_dup_node_elems(elem_tet)
    self.node_tet=node_tet
    self.elem_tet=elem_tet
    self.face_tet=face_tet
    if eigenanalysis:
      self.eig_thread=CustomThread(target=run_nastran_eig,args=(elem_tet,node_tet,self.nastran_exe_path))
      self.eig_thread.start()
    mat_weight,self.nid_surf_tet=mapping_surfacenode_fast(stop_gradient(connect_ls),stop_gradient(coord_ls),node_tet,face_tet)
    
    self.set_target()
    self.dim_spc=nid2dim_3d(jnp.where(node_tet[:,1]==0.0)[0])
    coord_fem=reconstruct_coordinates(coord_ls,jnp.array(self.node_tet),self.nid_surf_tet,mat_weight)
    matKg,matMg=gmat_preprocess(self.elem_tet,coord_fem,self.young,self.poisson,self.rho)
    self.matKg=matKg; self.matMg=matMg
    self.loss_overhang=loss_overhang(connect_ls,coord_ls,self.angle_overhang,self.axis_overhang)
    return matKg,matMg
  
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
    msg_main =f'\nEpoch : {self.epoch}'
    msg_main+=f'\nloss (static) : {stop_gradient(loss_static):.3f}'
    msg_main+=f'\nloss (eigval) : {stop_gradient(loss_eigval):.3f} ({stop_gradient(v[0]):.3f})'
    msg_main+=f'\nloss (overhang) : {stop_gradient(self.loss_overhang):.3f}'
    msg_main+=f'\nloss (edge) : {stop_gradient(self.loss_edge):.3f}'
    fmt='{:.2f}  '*(v.shape[0]-1)
    msg_main+=('\nratio: '+fmt.format(*stop_gradient(v/v[0])[1:]))
    msg_main+=('\n ref : '+fmt.format(*(v_trg/v_trg[0])[1:]))
    fmt='{:.4f}  '*aux.shape[0]
    msg_main+=('\n cos : '+fmt.format(*aux)+'\n')
    self.logger_main.info(msg_main)

    loss=jnp.asarray([loss_static,loss_eigval,self.loss_overhang,self.loss_edge])
    return loss,loss
  
  def get_phi0_shell_open(self,n_msk=1):
    nid_root=np.where(self.coord_ls_str[:,1]==0.0)[0]
    _,count=np.unique(self.connect_ls_str,return_counts=True)
    nid_var_additional=np.intersect1d(np.where(count==4)[0],nid_root)
    self.nid_const=np.setdiff1d(self.nid_const,nid_var_additional)
    self.nid_var=np.union1d(self.nid_var,nid_var_additional)
    phi=init_phi_shell(self.coord_ls_str,self.nid_const,self.weightrbf,n_msk)
    return phi
  
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
  return np.asarray(stop_gradient(arr))

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

def ls_area(mesh3d):
  """
  Compute area of 3d mesh
  mesh3d : (n,3,3) float
  """
  v1=mesh3d[:,1]-mesh3d[:,0]
  v2=mesh3d[:,2]-mesh3d[:,0]
  cross=jnp.cross(v1,v2)
  area=jnp.linalg.norm(cross,axis=1).sum()/2.0
  return area

def ls_overhang_area(connect,coord,angle,axis=np.array([0.,1.0,0.])):
  """
  Compute area of overhang
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
  msk=cosines<-np.cos(np.deg2rad(angle))
  return area[msk].sum()

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