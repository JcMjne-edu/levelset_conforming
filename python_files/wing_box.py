"""
dat files for 2D wing can be found in the following link:
http://airfoiltools.com/search/index
"""

import numpy as np
import math
import plotly.graph_objects as go
from stl import mesh
import triangle as tr

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
  def __init__(self,wing2d:Wing2D,semispan,root_chord,tip_chord,
               sweep,locs_spar,locs_lib,
               ref_span_elem_scale=0.06,n_sep_spar=4):
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

  def export_marc(self,f_name,thickness,E=7e4,poisson=0.3,rho=2.7e-9,remove_edge=True):
    """
    Export skin mesh to marc input file
    """
    if not remove_edge:
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
    else:
      ids=np.where(self.points[:,0]==self.locs_spar[-1])[0]
      _connect_skin=self.connect_skin.reshape(self.ny-1,-1,4)
      ids=np.arange(ids[0],ids[1])
      _ids=ids.repeat(self.ny-1).reshape(-1,self.ny-1) #(m,ny-1)
      _ids=_ids+np.arange(self.ny-1)*self.points.shape[0]-np.arange(self.ny-1) #(m,ny-1)
      _connect_skin=self.connect_skin[_ids.flatten()] #_connect_skin[:,ids[0]:ids[1]-1].reshape(-1,4)
      _connect_quad=np.concatenate([_connect_skin,self.connect_spar],axis=0)
      _connect_tri=self.connect_lib.reshape(self.n_lib,-1,3)[:,:self.n_tri_lib].reshape(-1,3) #(n_face2,3)

      _connect_lib=self.connect_lib.reshape(self.n_lib,-1,3)[:,:self.n_tri_lib]
      _connect_spar_last=self.connect_spar.reshape(self.n_spar,-1)[-1]
      surface_nid=np.unique(np.concatenate([_connect_skin.flatten(),
                                            _connect_spar_last.flatten(),
                                            _connect_lib[0].flatten(),
                                            _connect_lib[-1].flatten()]))
      surface_connect_quad=np.concatenate([_connect_skin,_connect_spar_last.reshape(-1,4)[:,::-1]])
      surface_connect_tri=np.concatenate([_connect_lib[0],_connect_lib[-1]])
      surface_connect_tri=np.concatenate([surface_connect_tri,surface_connect_quad[:,:3],surface_connect_quad[:,np.array([0,2,3])]])
      
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
    
    marcmodel=to_marcinput_shell(self)
    marcmodel.add_material(E,poisson,rho,thickness)
    marcmodel.set_nmode(6)
    marcmodel.write(f_name)
      
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

def _connect_spar(coord_spar):
  """
  Input
  -----
  coord_spar : (n_spar,2,ny,3)
  """
  n_spar=coord_spar.shape[0]; ny=coord_spar.shape[2]
  _X=np.arange(ny); _Y=np.arange(n_spar)
  _X,_Y=np.meshgrid(_X[:-1],_Y) #(nY,nX)
  _X=_X.flatten(); _Y=_Y.flatten()
  idx1=_X+_Y*2*ny #(nX*nY,)
  idx2=idx1+1; idx3=idx2+ny; idx4=idx3-1 #(nX*nY,)
  idx=np.vstack((idx1,idx2,idx3,idx4)).T #(~,4)
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

class to_marcinput_shell:
  def __init__(self,model:Wing3D,title='shell',ldname='loadcase1',alloc=100,post_increment=1):
    self.title=title
    self.ldname=ldname
    self.alloc=alloc
    self.model=model
    self.post_increment=post_increment
    self._add_spc()
  
  def set_nmode(self,num_modes=6):
    self.num_modes=num_modes
  
  def write(self,path):
    self.nelem=self.model.connect_quad.shape[0]+self.model.connect_tri.shape[0]
    self.nnode=self.model.coordinates.shape[0]
    texts=[]
    texts.append(f'TITLE               {self.title},')
    texts.append('EXTENDED,')
    texts.append(f'SIZING, 0, {self.nelem}, {self.nnode}, 0')
    texts.append(f'ALLOCATE, {self.alloc},')
    texts.append(f'ELEMENTS, 139, 138')
    texts.append('VERSION, 13,')
    texts.append('TABLE, 0, 0, 2, 1, 1, 0, 0, 1')
    texts.append('PROCESSOR, 1, 1, 1, 0')
    texts.append('$NO LIST,')
    texts.append(f'DYNAMIC, 1, {self.num_modes}, 1, 0, 0, 0, 0, 0, 0')
    texts.append('ALL POINTS,')
    texts.append('LUMP, 1, 0')
    texts.append('PRINT, 1')
    texts.append('NO ECHO, 1, 2, 3')
    texts.append('SHELL SECT, 5, 0, 1')
    texts.append(f'END,')
    texts.append('SOLVER,\n 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')
    texts.append('OPTIMIZE, 11')
    self._write_connectivity(texts)
    self._write_coordinates(texts)
    self._write_spc(texts)
    self._write_material(texts)
    self._write_geometry(texts)
    self._write_fixed_disp(texts)
    self._write_loadcase(texts)
    texts.append('NO PRINT,')
    #self._write_print_node(f)
    self._write_post(texts)
    self._write_parameters(texts)
    texts.append('END OPTION,')
    texts.append(f'TITLE, {self.ldname}')
    self._write_loadcase_his(texts)
    self._write_control(texts)
    self._write_modal_shape(texts)
    texts.append('CONTINUE,')
    self._write_recover(texts)
    texts.append('CONTINUE,\n')
    with open(path,mode='w') as f:
      f.write('\n'.join(texts))

  def _write_connectivity(self,texts):
    texts.append('CONNECTIVITY,')
    texts.append(f' 0, 0, 1, 0, 1, 1, 0, 0, 0')
    nelem=self.model.connect_quad.shape[0]+self.model.connect_tri.shape[0]
    nnode=self.model.coordinates.shape[0]
    self.nnode=nnode
    order_nelements=int(np.log10(nelem))+1
    order_nnodes=int(np.log10(nnode))+1
    def _text_connect(connect,eid,i_elem):
      arg=f' {eid:{order_nelements}}, {i_elem}'
      for j in range(len(connect)):
        arg+=f', {connect[j]:{order_nnodes}}'
      return arg
    for i,connect in enumerate(self.model.connect_quad):
      texts.append(_text_connect(connect+1,i+1,139))
    n_quad=self.model.connect_quad.shape[0]
    for i,connect in enumerate(self.model.connect_tri):
      texts.append(_text_connect(connect+1,i+1+n_quad,138))
    
  def _write_coordinates(self,texts):
    texts.append('COORDINATES,')
    texts.append(f' 3, {self.nnode}, 0, 1')
    order_nnodes=int(np.log10(self.nnode))+1
    order_c=int(np.log10(self.model.coordinates.max()))+2
    nf=4
    for idx,(x,y,z) in enumerate(self.model.coordinates):
      texts.append(f' {int(idx+1):{order_nnodes}}, {x:{order_c+nf}.5f}, {y:{order_c+nf}.5f}, {z:{order_c+nf}.5f}')

  def _add_spc(self):
    self.spc=np.where(self.model.coordinates[:,1]==0.0)[0]+1 #(nspc,)

  def _write_spc(self,texts):
    texts.append('DEFINE, NODE, SET, spc_nodes')
    arg=''
    for i,spcnode in enumerate(self.spc):
      arg+=f'{spcnode}, '
      if i%13==12 and i!=len(self.spc)-1:
        arg+='c\n'
    texts.append(arg)

  def _write_geometry(self,texts):
    """
    -------------------
    """
    texts.append('GEOMETRY,')
    texts.append('0, 0, 2')
    texts.append('1, 8')
    texts.append('geom1,')
    texts.append(f'{self.thickness}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n')

  def _write_fixed_disp(self,texts):
    """
    ------------------------
    """
    texts.append('FIXED DISP,\n')
    texts.append('1, 0, 0, 0, 1, 0, apply1')
    texts.append('0.0, 0.0, 0.0')
    texts.append('0, 0, 0')
    texts.append('1, 2, 3')
    texts.append('2,')
    texts.append('spc_nodes,')

  def add_material(self,young,poisson,density,thickness):
    self.young=young
    self.poisson=poisson
    self.density=density
    self.thickness=thickness
  
  def _write_material(self,texts):
    texts.append('ISOTROPIC,\n')
    texts.append('1, ELASTIC, ISOTROPIC, 10, 0, 0, 0, material1')
    texts.append(f'{self.young}, {self.poisson}, {self.density}, 0.0, 0.0, 0.0, 0.0, 0.0')
    texts.append('0, 0, 0, 0, 0, 0, 0, 0\n')

  def _write_loadcase(self,texts):
    texts.append('LOADCASE, job1')
    texts.append('1,')
    texts.append('apply1,')
  
  def _write_loadcase_his(self,texts):
    texts.append(f'LOADCASE, {self.ldname}')
    texts.append('1,')
    texts.append('apply1,')

  def _write_modal_shape(self,texts):
    texts.append('MODAL SHAPE,')
    texts.append(f'0.0, 0.0, {self.num_modes}, 0, 0, 0.0, 0.0, 0.0')

  def _write_recover(self,texts):
    texts.append('RECOVER,')
    texts.append(f'1, {self.num_modes}, 0')

  def _write_control(self,texts):
    texts.append('CONTROL,')
    texts.append('99999, 10, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0')
    texts.append('0.001 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0')
  
  def _write_parameters(self,texts):
    texts.append('PARAMETERS,')
    texts.append('1.0, 1.0E+9, 1.0E+2, 1.0E+6, 2.5E-1, 5.0E-1, 1.5, -5.0E-1')
    texts.append('8.625, 20.0, 1.0E-4, 1.0E-6, 1.0, 1.0E-4')
    texts.append('8.314, 2.7315E+2, 5.0E-1, 0.0, 5.67051E-8, 1.438769E-2, 2.9979E+8, 1.0E+30')
    texts.append('0.0, 0.0, 1.0E+2, 0.0, 1.0, -2.0, 1.0E+6, 3.0')
    texts.append('0.0, 0.0, 1.256637061E-6, 8.854187817E-12, 1.2E+2, 1.0E-3, 1.6E+2, 0.0')
    texts.append('3.0, 4.0E-1')

  def _write_post(self,texts):
    texts.append('POST,')
    texts.append(f'2, 16, 17, 0, 0, 19, 20, 0, {self.post_increment}, 0, 0, 0, 0, 0, 0, 0')
    texts.append('311, 0')
    texts.append('401, 0')

  def _write_print_node(self,texts):
    texts.append('PRINT NODE,')
    texts.append('3,1,6,0')
    texts.append('TOTA,')
    texts.append(f'{self.nnodes-2} TO {self.nnodes}')
    texts.append('PRINT ELEMENT,\n\n\n\n')