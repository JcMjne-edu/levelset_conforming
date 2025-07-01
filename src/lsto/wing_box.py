"""
dat files for 2D wing can be found in the following link:
http://airfoiltools.com/search/index
"""

import numpy as np
import math
import plotly.graph_objects as go
from stl import mesh
import triangle as tr
from pyNastran.bdf.bdf import BDF,CaseControlDeck

class Wing2D:
  """
  Read wing data from dat file
  """
  def __init__(self,f_name_dat,close=False):
    f=open(f_name_dat,'r')
    data=f.read().splitlines()
    f.close()
    data=data[1:]
    points=[]
    for line in data:
      points.append([float(line.split()[0]),float(line.split()[1])])
    self.points=np.array(points) #(num_points,2)
    if close:
      self.points=np.vstack((self.points,self.points[0,:]))

class Wing3D:
  def __init__(self,wing2d:Wing2D,semispan,root_chord,tip_chord,sweep,locs_spar,
               locs_lib,ref_span_elem_scale=0.06,n_sep_spar=4):
    """
    Parameters
    ----------
    wing2d : Wing2D
    semispan : float
    root_chord : float
    tip_chord : float
    sweep : float (deg)
    n_sep_semispan : int
    locs_spar : float (0 to 1) list
    locs_lib : float (0 to 1) list
    """
    points=wing2d.points
    self.points2d=wing2d.points
    assert type(locs_spar)==list
    assert type(locs_lib)==list
    locs_spar.sort(); locs_lib.sort()
    assert locs_spar[0]>=0. and locs_spar[-1]<=1.
    assert locs_lib[0]>=0. and locs_lib[-1]<=1.
    
    points,spar_points=self._set_spar_points(locs_spar,points) #(n_spar,2,2)
    self.points2d_spar=points
    n_spar=spar_points.shape[0]
    n_points_2d=points.shape[0]
    y_span,ny,id_libs=self._set_loc_span(locs_lib,ref_span_elem_scale,semispan) #(ny,)
    y=y_span.repeat(n_points_2d).reshape(ny,n_points_2d) #(ny,n_points2)
    xz=np.tile(points,(ny,1,1)) #(ny,num_points2,2)
    xyz=np.array((xz[:,:,0],y,xz[:,:,1])).transpose(1,2,0) #(ny,n_points2,3)
    connect_skin=_connect_skin(xyz) #((np-1)*(ny-1),4)
    chord=np.linspace(root_chord,tip_chord,ny) #(ny,)
    sweep=sweep*np.pi/180. #(rad)
    sin_sweep=np.sin(sweep)
    offset_x=y_span*sin_sweep+0.25*root_chord-0.25*chord #(ny,)
    offset=np.array((offset_x,np.zeros_like(offset_x),np.zeros_like(offset_x))).T #(ny,3)
    scale=np.array((chord,np.ones_like(chord),chord)).T #(ny,3)
    coord_skin=xyz*scale[:,None]+offset[:,None] #(ny,n_points2,3)
    self.coord_skin=coord_skin.reshape(-1,3) #((ny)*n_points2,3)
    self.connect_skin=connect_skin #((np-1)*(ny-1),4)
    self.ny=ny
    self.points=points

    #set spar mesh
    coord_spar_y=y_span.repeat(n_spar).reshape(-1,n_spar).T #(n_spar,ny)
    coord_spars=[]
    for i in range(2):
      coord_spar_x=spar_points[:,i,0].repeat(ny).reshape(n_spar,-1) #(n_spar,ny)
      coord_spar_z=spar_points[:,i,1].repeat(ny).reshape(n_spar,-1) #(n_spar,ny)
      scale=np.linspace(root_chord,tip_chord,ny) #(ny,)
      coord_spar_x=coord_spar_x*scale+offset_x; coord_spar_z=coord_spar_z*scale #(n_spar,ny)
      coord_spars.append(np.array((coord_spar_x,coord_spar_y,coord_spar_z)).transpose(1,2,0)) #(n_spar,ny,3)
    coord_spar=np.array(coord_spars).transpose(1,0,2,3) #(n_spar,2,ny,3)
    msk_subdivide=np.array([np.arange(0,n_sep_spar+1),np.arange(0,n_sep_spar+1)[::-1]])/n_sep_spar #(2,n_sep_spar+1) 
    coord_spar=np.einsum('ij,kilm->kjlm',msk_subdivide,coord_spar) #(n_spar,n_sep_spar+1,ny,3)
    self.connect_spar=_connect_spar_sep(coord_spar) 
    self.coord_spar=coord_spar.reshape(-1,3) #(n_spar*ny*2,3)
    i_offset_spar=self.coord_skin.shape[0]
    self.connect_spar=self.connect_spar+i_offset_spar #(n_spar*(ny-1),4)
    #set lib mesh
    coord_lib=[]
    connect_lib=[]
    coords=np.concatenate([self.coord_skin,self.coord_spar],axis=0) #(n_node,3)
    starting_nid=i_offset_spar+self.coord_spar.shape[0]
    self.connects_spar=[]
    #_id_libs=np.hstack([id_libs,])
    self.n_lib=len(id_libs)
    self.locs_spar=locs_spar
    for i in range(len(id_libs)):
      _coord,_connect=self._spar_id(n_sep_spar,i_offset_spar,
      id_libs[i],coords,starting_nid+i*(len(locs_spar)))
      coord_lib.append(_coord); connect_lib.append(_connect)
      self.connects_spar.append(_connect)
    self.coord_lib=np.concatenate(coord_lib,axis=0) 
    self.connect_lib=np.concatenate(connect_lib,axis=0) 
    self.n_spar=n_spar
    self.root_chord=root_chord
    self.tip_chord=tip_chord
    self.sin_sweep=sin_sweep
    self.semispan=semispan
    self.p1aero=np.array([0.,0.,0.])
    self.p2aero=np.array([root_chord,0.,0.])
    self.p3aero=np.array([offset_x[-1],semispan,0.0])
    self.p4aero=np.array([offset_x[-1]+tip_chord,semispan,0.0])
    self._extract_surface()

  def _set_spar_points(self,locs_spar,points):
    """
    Return
    ------
    spar_points : (n_spar,2,2)
    """
    spar_points=[]
    for loc_spar in locs_spar:
      points,spar_point=self._spar(loc_spar,points)
      spar_points.append(spar_point)
    return points,np.array(spar_points)

  def _set_loc_span(self,locs_lib,ref_span_elem_scale,semispan):
    loc_inner_lib=0.0
    n_nodes=0
    _locs_lib=locs_lib+[1.0]
    y_list=[np.zeros(1)]
    id_libs=[]
    for loc_lib in _locs_lib:
      dist_libs=loc_lib-loc_inner_lib
      num_span_elem=math.ceil(dist_libs/ref_span_elem_scale)
      y_locs=np.linspace(loc_inner_lib,loc_lib,num_span_elem+1)[1:]
      y_list.append(y_locs)
      id_libs.append(max(y_locs.shape[0]+n_nodes,0))
      loc_inner_lib=loc_lib
      n_nodes=id_libs[-1]
    y_span=np.hstack(y_list)*semispan
    id_libs=np.array(id_libs)
    ny=y_span.shape[0]
    return y_span,ny,id_libs
  
  def _spar(self,loc_spar,points):
    """
    Parameters
    ----------
    loc_spar : float (0 to 1)
    points : (num_points,2)
    """
    if loc_spar in points[:,0]:
      ids=np.where(points[:,0]==loc_spar)[0]
      z1=points[ids[0],1]; z2=points[ids[1],1]
      spar_point=np.array([[loc_spar,z1],[loc_spar,z2]]) #(2,2)
      return points,spar_point
    msk=points[:,0]<loc_spar
    ind1=np.where(~msk[:-1]*msk[1:])[0][0]+1
    z1=((loc_spar-points[ind1-1,0])*points[ind1,1]+(points[ind1,0]-loc_spar)*points[ind1-1,1])/(points[ind1,0]-points[ind1-1,0])
    points=np.vstack((points[:ind1],np.array([loc_spar,z1]),points[ind1:])) #(num_points+1,2)
    msk=points[:,0]<loc_spar
    ind2=np.where(msk[:-1]*~msk[1:])[0][0]+1
    z2=((loc_spar-points[ind2-1,0])*points[ind2,1]+(points[ind2,0]-loc_spar)*points[ind2-1,1])/(points[ind2,0]-points[ind2-1,0])
    points=np.vstack((points[:ind2],np.array([loc_spar,z2]),points[ind2:])) #(num_points+2,2)
    spar_point=np.array([[loc_spar,z1],[loc_spar,z2]]) #(2,2)
    return points,spar_point
  
  def extract_surface(self):
    """
    Return
    ------
    surface_nid : (n_surface_node,)
    surface_connect_tri : (n_surface_face,3)
    """
    def extract_loop(seg):
      """
      seg : int (n,2)
      """
      loop=[seg[0,0]]
      start=seg[0,0]
      i1=0
      val=seg[i1,1]
      while True:
        if val==start:
          break
        loop.append(val)
        i2=np.where(seg==val)[0]
        i1=np.setdiff1d(i2,i1)
        vals=seg[i1]
        val=np.setdiff1d(vals,val)[0]
      return loop
    
    points=self.points2d
    seg=np.array([np.arange(points.shape[0]),np.arange(1,points.shape[0]+1)%points.shape[0]]).T
    seg_markers=np.zeros((seg.shape[0],1))
    dict_input= dict(vertices=points,segments=seg)
    triangulation=tr.triangulate(dict_input,'p')
    coord2d=triangulation['vertices'] #(n_points3,2)
    coord3d=np.array((coord2d[:,0],np.zeros_like(coord2d[:,0]),coord2d[:,1])).T #(n_points3,3)
    connect2d=triangulation['triangles']
    loop=extract_loop(triangulation['segments'])
    points=coord2d[loop[::-1]]

    n_points_2d=points.shape[0]
    y_span=np.array([0.,self.semispan]) #(2,)
    y=y_span.repeat(n_points_2d).reshape(2,n_points_2d) #(2,n_points2)
    xz=np.tile(points,(2,1,1)) #(2,num_points2,2)
    xyz=np.array((xz[:,:,0],y,xz[:,:,1])).transpose(1,2,0) #(2,n_points2,3)
    connect_skin=_connect_skin_full(xyz.shape[1]) #((np2-1),4)
    chord=np.array([self.root_chord,self.tip_chord]) #(2,)
    offset_x=y_span*self.sin_sweep+0.25*self.root_chord-0.25*chord #(2,)
    offset=np.array((offset_x,np.zeros_like(offset_x),np.zeros_like(offset_x))).T #(2,3)
    scale=np.array((chord,np.ones_like(chord),chord)).T #(2,3)
    coord_skin=(xyz*scale[:,None]+offset[:,None]).reshape(-1,3) #(2*n_points2,3)
    connect_skin_tri=np.concatenate([connect_skin[:,:3],connect_skin[:,[0,2,3]]],axis=0) #(n_face2,3)
    offset[1,1]=self.semispan
    coord_lib=(coord3d*scale[:,None]+offset[:,None]).reshape(-1,3) #(2*n_points2,3)
    connect_lib=np.concatenate([connect2d[:,::-1],connect2d+coord2d.shape[0]]) #(n_face3,3)
    connect_lib_outer=connect2d+coord2d.shape[0] #(n_face3,3)
    coordinates_surface=np.concatenate([coord_skin,coord_lib],axis=0) #(n_node_surface,3)
    connectivity_surface=np.concatenate([connect_skin_tri,connect_lib+coord_skin.shape[0]],axis=0)[:,::-1] #(n_face_surface,3)
    root_edge_nid=np.arange(y.shape[1])
    connectivity_surface_open=np.concatenate([connect_skin_tri,connect_lib_outer+coord_skin.shape[0]],axis=0)[:,::-1] #(n_face_surface,3)

    used_nid=np.unique(connectivity_surface)
    unused_nid=np.setdiff1d(np.arange(coordinates_surface.shape[0]),used_nid)
    coordinates_surface[unused_nid]=coordinates_surface[used_nid[0]]
    self.coordinates_surface,inverse=np.unique(coordinates_surface,axis=0,return_inverse=True)
    self.connectivity_surface=inverse[connectivity_surface]
    self.connectivity_surface_open=inverse[connectivity_surface_open]
    self.root_edge_nid=inverse[root_edge_nid]

  def _extract_surface(self):
    """
    Return
    ------
    surface_nid : (n_surface_node,)
    surface_connect_tri : (n_surface_face,3)
    """
    def extract_loop(seg):
      """
      seg : int (n,2)
      """
      loop=[seg[0,0]]
      start=seg[0,0]
      i1=0
      val=seg[i1,1]
      while True:
        if val==start:
          break
        loop.append(val)
        i2=np.where(seg==val)[0]
        i1=np.setdiff1d(i2,i1)
        vals=seg[i1]
        val=np.setdiff1d(vals,val)[0]
      return loop
    
    points=self.points2d_spar
    seg=np.array([np.arange(points.shape[0]),np.arange(1,points.shape[0]+1)%points.shape[0]]).T
    seg_markers=np.zeros((seg.shape[0],1))
    dict_input= dict(vertices=points,segments=seg)
    triangulation=tr.triangulate(dict_input,'p')
    coord2d=triangulation['vertices'] #(n_points3,2)
    coord3d=np.array((coord2d[:,0],np.zeros_like(coord2d[:,0]),coord2d[:,1])).T #(n_points3,3)
    connect2d=triangulation['triangles']
    loop=extract_loop(triangulation['segments'])
    points=coord2d[loop[::-1]]

    n_points_2d=points.shape[0]
    y_span=np.array([0.,self.semispan]) #(2,)
    y=y_span.repeat(n_points_2d).reshape(2,n_points_2d) #(2,n_points2)
    xz=np.tile(points,(2,1,1)) #(2,num_points2,2)
    xyz=np.array((xz[:,:,0],y,xz[:,:,1])).transpose(1,2,0) #(2,n_points2,3)
    connect_skin=_connect_skin_full(xyz.shape[1]) #((np2-1),4)
    chord=np.array([self.root_chord,self.tip_chord]) #(2,)
    offset_x=y_span*self.sin_sweep+0.25*self.root_chord-0.25*chord #(2,)
    offset=np.array((offset_x,np.zeros_like(offset_x),np.zeros_like(offset_x))).T #(2,3)
    scale=np.array((chord,np.ones_like(chord),chord)).T #(2,3)
    coord_skin=(xyz*scale[:,None]+offset[:,None]).reshape(-1,3) #(2*n_points2,3)
    connect_skin_tri=np.concatenate([connect_skin[:,:3],connect_skin[:,[0,2,3]]],axis=0) #(n_face2,3)
    offset[1,1]=self.semispan
    coord_lib=(coord3d*scale[:,None]+offset[:,None]).reshape(-1,3) #(2*n_points2,3)
    connect_lib=np.concatenate([connect2d[:,::-1],connect2d+coord2d.shape[0]]) #(n_face3,3)
    connect_lib_outer=connect2d+coord2d.shape[0] #(n_face3,3)
    coordinates_surface=np.concatenate([self.coord_skin,coord_skin,coord_lib],axis=0) #(n_node_surface,3)
    n1=self.coord_skin.shape[0]
    n2=n1+coord_skin.shape[0]
    _connect_skin_tri=self.connect_skin[:,[0,1,2,2,3,0]].reshape(-1,3)
    connectivity_surface=np.concatenate([_connect_skin_tri,connect_lib+n2],axis=0)[:,::-1] #(n_face_surface,3)
    root_edge_nid=np.arange(y.shape[1])
    connectivity_surface_open=np.concatenate([_connect_skin_tri,connect_lib_outer+n2],axis=0)[:,::-1] #(n_face_surface,3)


    used_nid=np.unique(connectivity_surface)
    unused_nid=np.setdiff1d(np.arange(coordinates_surface.shape[0]),used_nid)
    coordinates_surface[unused_nid]=coordinates_surface[used_nid[0]]
    self.coordinates_surface,inverse=np.unique(coordinates_surface,axis=0,return_inverse=True)
    self.connectivity_surface=inverse[connectivity_surface]
    self.connectivity_surface_open=inverse[connectivity_surface_open]
    self.root_edge_nid=inverse[root_edge_nid]

  def to_plotly(self):
    """
    Attributes
    ----------
    coordinates : (n_node,3)\n
    connect_skin : (n_tetmesh,4)\n
    coord_spar : (n_spar,ny,2,3)\n
    connect_spar : (n_spar*(ny-1),4)\n
    Return
    ------
    ind_skin : Arr float (n_trimesh1,3)\n
    spars : Arr float ((n_trimesh2,3,3),...)
    """
    crd=np.concatenate([self.coord_skin,self.coord_spar,self.coord_lib],axis=0) #(n_node,3)
    ids1_skin1=np.array((self.connect_skin[:,0],self.connect_skin[:,1],self.connect_skin[:,2])).T
    ids2_skin2=np.array((self.connect_skin[:,0],self.connect_skin[:,2],self.connect_skin[:,3])).T
    ids_skin=np.vstack((ids1_skin1,ids2_skin2)) #(2*n_tetmesh,3)
    ids1_spar1=np.array((self.connect_spar[:,0],self.connect_spar[:,1],self.connect_spar[:,2])).T
    ids2_spar2=np.array((self.connect_spar[:,0],self.connect_spar[:,2],self.connect_spar[:,3])).T
    ids_spar=np.vstack((ids1_spar1,ids2_spar2)) #(2*n_tetmesh,3)
    self.crd=crd
    fig=go.Figure()
    fig.add_trace(go.Mesh3d(x=crd[:,0],y=crd[:,1],z=crd[:,2],i=ids_skin[:,0],j=ids_skin[:,1],k=ids_skin[:,2],color='blue',opacity=0.3))
    fig.add_trace(go.Mesh3d(x=crd[:,0],y=crd[:,1],z=crd[:,2],i=ids_spar[:,0],j=ids_spar[:,1],k=ids_spar[:,2],color='red',opacity=0.3))
    fig.add_trace(go.Mesh3d(x=crd[:,0],y=crd[:,1],z=crd[:,2],i=self.connect_lib[:,0],j=self.connect_lib[:,1],k=self.connect_lib[:,2],color='green',opacity=0.3))
    fig.update_layout(scene_aspectmode='data')
    return fig
  
  def export_obj(self,f_name):
    """
    Export skin mesh to obj file

    """
    _coords=np.concatenate([self.coord_skin,self.coord_spar,self.coord_lib],axis=0) #(n_node,3)
    _connect_quad=np.concatenate([self.connect_skin,self.connect_spar],axis=0) #(n_face,4)
    _connect_tri=self.connect_lib #(n_face,3)
    unique_coords,inverse=np.unique(_coords,axis=0,return_inverse=True)
    self.coordinates=unique_coords
    self.connect_quad=inverse[_connect_quad]
    self.connect_tri=inverse[_connect_tri]
    texts=[]
    for vertex in self.coordinates:
      texts.append(f"v {' '.join(map(str, vertex))}")
    for face in self.connect_quad:
      face = [str(idx + 1) for idx in face]
      texts.append(f"f {' '.join(face)}")
    for face in self.connect_tri:
      face = [str(idx + 1) for idx in face]
      texts.append(f"f {' '.join(face)}")
    with open(f_name,'w') as f:
      f.write('\n'.join(texts))

  def export_nastran(self,fname1,fname2,thickness_root,thickness_tip,young,poisson,rho,num_modes=6):
    """
    Export skin mesh to nastran input file
    """
    _connect_quad=np.concatenate([self.connect_skin,self.connect_spar],axis=0) #(n_face,4)
    _connect_tri=self.connect_lib #(n_face,3)
    connect_lib=self.connect_lib.reshape(self.n_lib,-1,3) #(n_lib,n_face,3)
    surface_nid=np.unique(np.concatenate([self.connect_skin.flatten(),
                                          connect_lib[0].flatten(),
                                          connect_lib[-1].flatten()]))
    surface_connect_quad=self.connect_skin
    surface_connect_tri=np.concatenate([connect_lib[0],connect_lib[-1]])
    surface_connect_tri=np.concatenate([surface_connect_tri,surface_connect_quad[:,:3],surface_connect_quad[:,np.array([0,2,3])]])
    _connect_skin=self.connect_skin
      
    _coords=np.concatenate([self.coord_skin,self.coord_spar,self.coord_lib],axis=0) #(n_node,3)
    nid_used=np.unique(np.concatenate((_connect_quad.flatten(),_connect_tri.flatten())))
    nid_not_used=np.setdiff1d(np.arange(_coords.shape[0]),nid_used)
    _coords[nid_not_used]=_coords[nid_used[0]]
    unique_coords,inverse=np.unique(_coords,axis=0,return_inverse=True)
    self.coordinates=unique_coords
    self.connect_quad=inverse[_connect_quad]
    self.connect_skin_fem=inverse[_connect_skin]
    self.connect_tri=inverse[_connect_tri]
    self.surface_nid=np.unique(inverse[surface_nid])
    self.surface_connect_tri=inverse[surface_connect_tri]
    
    to_nastraninput_shell(self,num_modes,young,poisson,rho,thickness_root,thickness_tip,fname1,fname2)

  def export_nastran_aero(self,fname,thickness_root,thickness_tip,nx,ny,vmin,vmax,
                          young,poisson,rho,rho_air=None,num_modes=None,
                          nvelocity=None,flutter=True,q=None,anglea=None):
    _connect_quad=np.concatenate([self.connect_skin,self.connect_spar],axis=0) #(n_face,4)
    _connect_tri=self.connect_lib #(n_face,3)
    connect_lib=self.connect_lib.reshape(self.n_lib,-1,3) #(n_lib,n_face,3)
    surface_nid=np.unique(np.concatenate([self.connect_skin.flatten(),
                                          connect_lib[0].flatten(),
                                          connect_lib[-1].flatten()]))
    surface_connect_quad=self.connect_skin
    surface_connect_tri=np.concatenate([connect_lib[0],connect_lib[-1]])
    surface_connect_tri=np.concatenate([surface_connect_tri,surface_connect_quad[:,:3],surface_connect_quad[:,np.array([0,2,3])]])
    _connect_skin=self.connect_skin
      
    _coords=np.concatenate([self.coord_skin,self.coord_spar,self.coord_lib],axis=0) #(n_node,3)
    nid_used=np.unique(np.concatenate((_connect_quad.flatten(),_connect_tri.flatten())))
    nid_not_used=np.setdiff1d(np.arange(_coords.shape[0]),nid_used)
    _coords[nid_not_used]=_coords[nid_used[0]]
    unique_coords,inverse=np.unique(_coords,axis=0,return_inverse=True)
    self.coordinates=unique_coords
    self.connect_quad=inverse[_connect_quad]
    self.connect_skin_fem=inverse[_connect_skin]
    self.connect_tri=inverse[_connect_tri]
    self.surface_nid=np.unique(inverse[surface_nid])
    self.surface_connect_tri=inverse[surface_connect_tri]
    
    if flutter:
      to_nastraninput_shell_145(self,num_modes,young,poisson,rho,rho_air,thickness_root,
                                thickness_tip,fname,nx,ny,vmin,vmax,nvelocity)
    else:
      to_nastraninput_shell_144(self,young,poisson,rho,q,thickness_root,
                                thickness_tip,fname,nx,ny,anglea)
      
  def export_stl(self,f_name,remove_edge=True):
    """
    Export skin mesh to stl file
    using connectivity data (self.connect_tri) and coordinate data (self.coordinates),
    create stl file using numpy-stl package
    """
    if not remove_edge:
      connect_lib=self.connect_lib.reshape(self.n_lib,-1,3) #(n_lib,n_face,3)
      connect_skin1=self.connect_skin[:,:3] #(n_face,3)
      connect_skin2=np.array((self.connect_skin[:,0],self.connect_skin[:,2],self.connect_skin[:,3])).T #(n_face,3)
      connect_tri=np.vstack((connect_skin1,connect_skin2,connect_lib[0,:,::-1],connect_lib[-1])) #(n_face2,3)
      mesh_data=mesh.Mesh(np.zeros(connect_tri.shape[0],dtype=mesh.Mesh.dtype))
      for i, f in enumerate(connect_tri):
        for j in range(3):
          mesh_data.vectors[i][j]=self.crd[f[j],:]
      mesh_data.save(f_name)
      self.connect_tri_stl=connect_tri
      return mesh_data
    else:
      ids=np.where(self.points[:,0]==self.locs_spar[-1])[0]
      _connect_skin=self.connect_skin.reshape(self.ny-1,-1,4)
      ids=np.arange(ids[0],ids[1])
      _ids=ids.repeat(self.ny-1).reshape(-1,self.ny-1) #(m,ny-1)
      _ids=_ids+np.arange(self.ny-1)*self.points.shape[0]-np.arange(self.ny-1) #(m,ny-1)

      _connect_skin=self.connect_skin[_ids.flatten()] #_connect_skin[:,ids[0]:ids[1]-1].reshape(-1,4)
      _connect_spar=self.connect_spar.reshape(len(self.locs_spar),-1,4)[-1,:,::-1]
      _connect_lib=self.connect_lib.reshape(self.n_lib,-1,3)[:,:self.n_tri_lib]
      _connect_quad=np.concatenate([_connect_skin,_connect_spar],axis=0)
      _connect_tri1=np.array((_connect_quad[:,0],_connect_quad[:,1],_connect_quad[:,2])).T
      _connect_tri2=np.array((_connect_quad[:,0],_connect_quad[:,2],_connect_quad[:,3])).T
      _connect_tri=np.vstack((_connect_tri1,_connect_tri2,_connect_lib[0,:,::-1],_connect_lib[-1])) #(n_face2,3)
      mesh_data=mesh.Mesh(np.zeros(_connect_tri.shape[0],dtype=mesh.Mesh.dtype))
      for i, f in enumerate(_connect_tri):
        for j in range(3):
          mesh_data.vectors[i][j]=self.crd[f[j],:]
      mesh_data.save(f_name)
      self.connect_tri_stl=_connect_tri
      return mesh_data
  
  def _spar_id(self,n_sep_spar,i_offset_spar,
             id_lib,coords,starting_nid,if_cap=True):
    """
    Parameters
    ----------
    points : (n_p,2)\n
    coords : (n_node,3)\n
    Return
    ------
    lib_id_front : (~,3)
    """
    n_p=self.points.shape[0]
    i_center=starting_nid
    coord=[]
    connect=[]
    n_tri=0
    for i,loc_spar in enumerate(self.locs_spar):
      ids=np.where(self.points[:,0]==loc_spar)[0]
      if i==0:
        id_loop=np.hstack([np.arange(ids[0],ids[1]+1)+n_p*id_lib,
                          np.arange(1,n_sep_spar)*self.ny+i_offset_spar+id_lib])
      else:
        id_loop=np.hstack([np.arange(ids[0],ids_last[0]+1)+n_p*id_lib,
                          (np.arange(n_sep_spar-1)+i)[::-1]*self.ny+self.ny*n_sep_spar*(i-1)+i_offset_spar+id_lib,
                          np.arange(ids_last[1],ids[1]+1)+n_p*id_lib,
                          (np.arange(n_sep_spar-1)+i+1)*self.ny+self.ny*n_sep_spar*i+i_offset_spar+id_lib])
      
      v_center=np.mean(coords[id_loop],axis=0) #(3,)
      coord.append(v_center)
      f_loop=_fill_loop(id_loop,i_center)
      connect.append(f_loop)
      i_center+=1
      ids_last=ids
      n_tri+=f_loop.shape[0]
    self.n_tri_lib=n_tri
    if if_cap:
      id_loop=np.hstack([np.arange(ids[1],n_p)+n_p*id_lib,
                        np.arange(ids[0]+1)+n_p*id_lib,
                        (np.arange(n_sep_spar)+i)[::-1]*self.ny+self.ny*n_sep_spar*i+i_offset_spar+id_lib])
      n_lower=n_p-ids[1]
      n_upper=ids[0]+1
      offset=n_p*id_lib
      if n_lower>n_upper:
        for j in range(n_upper-2):
          connect.append([np.array([j,j+1,n_p-j-2])+offset,
                          np.array([j,n_p-j-2,n_p-j-1])+offset])
        i1=(np.arange(1,n_sep_spar+1)+i)[::-1]*self.ny+self.ny*n_sep_spar*i+i_offset_spar+id_lib
        i2=(np.arange(n_sep_spar)+i)[::-1]*self.ny+self.ny*n_sep_spar*i+i_offset_spar+id_lib
        _connect1=np.array([np.ones_like(i1)*(n_upper-2+offset),i1,i2]).T #(n_sep_spar-1,3)
        i1=(np.arange(n_p-n_lower,n_p-n_upper+1)+offset)
        i2=(np.arange(n_p-n_lower+1,n_p-n_upper+2)+offset)
        _connect2=np.array([np.ones_like(i1)*(n_upper-2+offset),i1,i2]).T #(n_upper-1,3)
        connect.append(_connect1); 
        connect.append(_connect2)
      else:
        raise NotImplementedError('n_lower<=n_upper is not implemented')
      v_center=np.mean(coords[id_loop],axis=0) #(3,)
      #coord.append(v_center)
      #connect.append(_fill_loop(id_loop,i_center))
    coord=np.array(coord) #(n_spar,3)
    connect=np.concatenate(connect,axis=0) #(n_p*n_spar,3)
    return coord,connect

def _connect_skin(_xyz):
  """
  Parameters
  ----------
  _xyz : (ny,nxz,3)
  Return
  ------
  idx : (~,4)
  """
  nxz=_xyz.shape[1]; ny=_xyz.shape[0]
  _X=np.arange(nxz); _Y=np.arange(ny)
  _X,_Y=np.meshgrid(_X[:-1],_Y[:-1]) #(nY,nX)
  _X=_X.flatten(); _Y=_Y.flatten()
  idx1=_X+_Y*nxz #(nX*nY,)
  idx2=idx1+1; idx3=idx2+nxz; idx4=idx3-1 #(nX*nY,)
  idx=np.vstack((idx1,idx2,idx3,idx4)).T #(~,4)
  idx_e1=np.arange(ny-1)*nxz; idx_e2=idx_e1+nxz
  idx_e3=np.arange(1,ny)*nxz+nxz-1; idx_e4=idx_e3-nxz
  idx_e=np.vstack((idx_e1,idx_e2,idx_e3,idx_e4)).T #((ny-1),4)
  idx=np.vstack((idx,idx_e)) #((ny-1)*nX,4)
  return idx

def _connect_skin_full(nxz):
  """
  Parameters
  ----------
  _xyz : (ny,nxz,3)
  Return
  ------
  idx : (~,4)
  """
  #nxz=_xyz.shape[1]; ny=_xyz.shape[0]
  _X=np.arange(nxz); _Y=np.arange(2)
  _X,_Y=np.meshgrid(_X[:-1],_Y[:-1]) #(nY,nX)
  _X=_X.flatten(); _Y=_Y.flatten()
  idx1=_X+_Y*nxz #(nX*nY,)
  idx2=idx1+1; idx3=idx2+nxz; idx4=idx3-1 #(nX*nY,)
  idx=np.vstack((idx1,idx2,idx3,idx4)).T #(~,4)
  idx_e1=nxz-1; idx_e2=0
  idx_e3=nxz; idx_e4=2*nxz-1
  idx_e=np.array((idx_e1,idx_e2,idx_e3,idx_e4)).T #((ny-1),4)
  idx=np.vstack((idx,idx_e)) #((ny-1)*nX,4)
  return idx
  
def _connect_spar_sep(coord_spar_sep):
  """
  coord_spar_sep : (n_spar,n_sep_spar+1,ny,3)
  """
  n_spar=coord_spar_sep.shape[0]
  n_sep_node=coord_spar_sep.shape[1]
  ny=coord_spar_sep.shape[2]
  _X=np.arange(ny)
  _Y=np.arange(n_spar*n_sep_node).reshape(n_spar,n_sep_node)
  _Y=np.hstack((_Y[:,:-1].flatten(),_Y[-1,-1])) #(n_spar*n_sep_node,)
  
  #_X=np.arange(nxz); _Y=np.arange(ny)
  _X,_Y=np.meshgrid(_X[:-1],_Y[:-1]) #(nY,nX)
  _X=_X.flatten(); _Y=_Y.flatten()
  idx1=_X+_Y*ny #(nX*nY,)
  idx2=idx1+1; idx3=idx2+ny; idx4=idx3-1 #(nX*nY,)
  idx=np.vstack((idx1,idx2,idx3,idx4)).T #(~,4)
  return idx
  
def _fill_loop(nodes,i_center):
  """
  nodes : Arr int (n,)\n
  i_center : int\n
  Return : 
  -----
  tri_connect : Arr int (n,3)
  """
  n=nodes.shape[0]
  connect1=np.ones(n,int)*i_center
  connect2=nodes[np.arange(n)-1]
  connect3=nodes[np.arange(n)]
  tri_connect=np.vstack((connect1,connect2,connect3)).T #(n,3)
  return tri_connect

def to_nastraninput_shell(model:Wing3D,num_modes,young,poisson,rho,thickness_root,thickness_tip,fname1,fname2):
  bdf=BDF(debug=None)
  bdf.sol=103
  cc=CaseControlDeck([
    'AUTOSPC(NOPRINT)=YES','DISP(PLOT,NORPRINT)=ALL','ECHO=NONE',
    'METHOD=100','SPC=1','TITLE=EIGVAL ANALYSIS MODEL','WEIGHTCHECK = YES'
    ])
  bdf.case_control_deck=cc
  bdf.add_param('POST',-1)
  bdf.add_eigrl(100,nd=num_modes)
  bdf.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)

  #add nodes
  for nid,coord in enumerate(model.coordinates):
    bdf.add_grid(nid+1,coord)
  #add elements
  center_quad=model.coordinates[model.connect_quad].mean(axis=1)
  thicknesses_quad=(thickness_root*(model.semispan-center_quad[:,1])+thickness_tip*center_quad[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_quad):
    bdf.add_cquad4(eid+1,pid=eid+1,nids=connect+1)
    bdf.add_pshell(pid=eid+1,mid1=1,t=thicknesses_quad[eid],mid2=1,mid3=1)
  n_quad=model.connect_quad.shape[0]
  center_tri=model.coordinates[model.connect_tri].mean(axis=1)
  thicknesses_tri=(thickness_root*(model.semispan-center_tri[:,1])+thickness_tip*center_tri[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_tri):
    bdf.add_ctria3(eid+1+n_quad,pid=eid+1+n_quad,nids=connect+1)
    bdf.add_pshell(pid=eid+1+n_quad,mid1=1,t=thicknesses_tri[eid],mid2=1,mid3=1)
  
  #add spc
  nid_spc=np.where(model.coordinates[:,1]==0.0)[0]+1
  bdf.add_spc1(conid=1,components='123456',nodes=nid_spc)
  bdf.write_bdf(fname1,write_header=False,interspersed=False)

  #-------------------
  # BDF for matrices
  bdf=BDF(debug=None)
  bdf.sol=103
  cc=CaseControlDeck([
    'ECHO=NONE','METHOD=100','SPC=1','TITLE=EIGVAL ANALYSIS MODEL'])
  bdf.case_control_deck=cc
  bdf.add_param('POST',-1)
  bdf.add_param('EXTOUT','DMIGPCH')
  bdf.add_eigrl(100,nd=1)
  bdf.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)

  #add nodes
  for nid,coord in enumerate(model.coordinates):
    bdf.add_grid(nid+1,coord)
  #add elements
  center_quad=model.coordinates[model.connect_quad].mean(axis=1)
  thicknesses_quad=(thickness_root*(model.semispan-center_quad[:,1])+thickness_tip*center_quad[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_quad):
    bdf.add_cquad4(eid+1,pid=eid+1,nids=connect+1)
    bdf.add_pshell(pid=eid+1,mid1=1,t=thicknesses_quad[eid],mid2=1,mid3=1)
  n_quad=model.connect_quad.shape[0]
  center_tri=model.coordinates[model.connect_tri].mean(axis=1)
  thicknesses_tri=(thickness_root*(model.semispan-center_tri[:,1])+thickness_tip*center_tri[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_tri):
    bdf.add_ctria3(eid+1+n_quad,pid=eid+1+n_quad,nids=connect+1)
    bdf.add_pshell(pid=eid+1+n_quad,mid1=1,t=thicknesses_tri[eid],mid2=1,mid3=1)
  
  #add spc
  nid_spc=np.where(model.coordinates[:,1]==0.0)[0]+1
  bdf.add_spc1(conid=1,components='123456',nodes=nid_spc)
  bdf.write_bdf(fname2,write_header=False,interspersed=False)

def to_nastraninput_shell_145(model:Wing3D,num_modes,young,poisson,rho,rho_air,
                               thickness_root,thickness_tip,fname,nx,ny,vmin,vmax,nvelocity):
  bdf=BDF(debug=None)
  bdf.sol=145
  cc=CaseControlDeck(['ECHO=NONE','METHOD=100','SPC=1','FMETHOD=40','SDAMP=2000'])
  bdf.case_control_deck=cc
  bdf.add_param('POST',-1)
  bdf.add_param('KDAMP',1)
  bdf.add_param('LMODES',num_modes)
  bdf.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)
  bdf.add_tabdmp1(2000,x=[0.0,10.0],y=[0.0,0.0])
  bdf.add_aero(None,cref=model.root_chord/2,rho_ref=rho_air)
  bdf.add_eigrl(100,nd=num_modes,norm='MAX')
  bdf.add_mkaero1(machs=[0.0,],reduced_freqs=[0.001,2.5])
  bdf.add_flutter(40,method='PK',density=1,mach=2,reduced_freq_velocity=4,imethod='S')
  bdf.add_flfact(1,[1.])
  bdf.add_flfact(2,[0.0])
  bdf.add_flfact(4,np.linspace(vmin,vmax,nvelocity))
  p4_x=model.sin_sweep*model.semispan+0.25*model.root_chord-0.25*model.tip_chord
  p4_y=model.semispan
  bdf.add_caero1(1,1,1,p1=[0.,0.,0.],x12=model.root_chord,p4=[p4_x,p4_y,0.0],
                 x43=model.tip_chord,nchord=nx,nspan=ny)
  bdf.add_paero1(1)
  bdf.add_spline1(1,1,1,nx*ny,1,1.0)

  vert=model.coordinates[model.connect_quad]
  norm=np.cross(vert[:,1]-vert[:,0],vert[:,2]-vert[:,0])
  norm=norm/np.linalg.norm(norm,axis=1,keepdims=True)
  msk=(norm[:,2]>0.7)
  nids=np.unique(model.connect_quad[msk])
  #nids=np.unique(model.connect_quad)
  #print(nids.shape)
  bdf.add_set1(1,nids+1)

  #add nodes
  for nid,coord in enumerate(model.coordinates):
    bdf.add_grid(nid+1,coord)
  #add elements
  center_quad=model.coordinates[model.connect_quad].mean(axis=1)
  thicknesses_quad=(thickness_root*(model.semispan-center_quad[:,1])+thickness_tip*center_quad[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_quad):
    bdf.add_cquad4(eid+1,pid=eid+1,nids=connect+1)
    bdf.add_pshell(pid=eid+1,mid1=1,t=thicknesses_quad[eid],mid2=1,mid3=1)
  n_quad=model.connect_quad.shape[0]
  center_tri=model.coordinates[model.connect_tri].mean(axis=1)
  thicknesses_tri=(thickness_root*(model.semispan-center_tri[:,1])+thickness_tip*center_tri[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_tri):
    bdf.add_ctria3(eid+1+n_quad,pid=eid+1+n_quad,nids=connect+1)
    bdf.add_pshell(pid=eid+1+n_quad,mid1=1,t=thicknesses_tri[eid],mid2=1,mid3=1)
  
  #add spc
  nid_spc=np.where(model.coordinates[:,1]==0.0)[0]+1
  bdf.add_spc1(conid=1,components='123456',nodes=nid_spc)
  bdf.write_bdf(fname,write_header=False,interspersed=False)

def to_nastraninput_shell_144(model:Wing3D,young,poisson,rho,q,
                               thickness_root,thickness_tip,fname,nx,ny,anglea):
  bdf=BDF(debug=None)
  bdf.sol=144
  cc=CaseControlDeck(['ECHO=NONE','SPC=1','TRIM=1','DISP=ALL','FORCE=ALL',
                      'AEROF=ALL','APRES=ALL','GPFO=ALL','STRESS=ALL',])
  bdf.case_control_deck=cc
  bdf.add_param('POST',-1)
  bdf.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)
  bdf.add_aestat(501,'ANGLEA')
  bdf.add_aestat(502,'PITCH')
  area=(model.root_chord+model.tip_chord)*model.semispan/2
  bdf.add_aeros(model.root_chord/2,2*model.semispan,area,0,0,1)
  bdf.add_trim(1,0.,q,['ANGLEA','PITCH'],[anglea,0.])
  p4_x=model.sin_sweep*model.semispan+0.25*model.root_chord-0.25*model.tip_chord
  p4_y=model.semispan
  bdf.add_caero1(1,1,1,p1=[0.,0.,0.],x12=model.root_chord,p4=[p4_x,p4_y,0.0],
                 x43=model.tip_chord,nchord=nx,nspan=ny)
  bdf.add_paero1(1)
  bdf.add_spline1(1,1,1,nx*ny,1,1.0)

  vert=model.coordinates[model.connect_quad]
  norm=np.cross(vert[:,1]-vert[:,0],vert[:,2]-vert[:,0])
  norm=norm/np.linalg.norm(norm,axis=1,keepdims=True)
  msk=(norm[:,2]>0.7)
  nids=np.unique(model.connect_quad[msk])
  #nids=np.unique(model.connect_quad)
  #print(nids.shape)
  bdf.add_set1(1,nids+1)

  #add nodes
  for nid,coord in enumerate(model.coordinates):
    bdf.add_grid(nid+1,coord)
  #add elements
  center_quad=model.coordinates[model.connect_quad].mean(axis=1)
  thicknesses_quad=(thickness_root*(model.semispan-center_quad[:,1])+thickness_tip*center_quad[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_quad):
    bdf.add_cquad4(eid+1,pid=eid+1,nids=connect+1)
    bdf.add_pshell(pid=eid+1,mid1=1,t=thicknesses_quad[eid],mid2=1,mid3=1)
  n_quad=model.connect_quad.shape[0]
  center_tri=model.coordinates[model.connect_tri].mean(axis=1)
  thicknesses_tri=(thickness_root*(model.semispan-center_tri[:,1])+thickness_tip*center_tri[:,1])/model.semispan
  for eid,connect in enumerate(model.connect_tri):
    bdf.add_ctria3(eid+1+n_quad,pid=eid+1+n_quad,nids=connect+1)
    bdf.add_pshell(pid=eid+1+n_quad,mid1=1,t=thicknesses_tri[eid],mid2=1,mid3=1)
  
  #add spc
  nid_spc=np.where(model.coordinates[:,1]==0.0)[0]+1
  bdf.add_spc1(conid=1,components='123456',nodes=nid_spc)
  bdf.write_bdf(fname,write_header=False,interspersed=False)


