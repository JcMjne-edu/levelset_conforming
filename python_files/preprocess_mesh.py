from jax import jit
from jax.experimental.sparse import BCOO,BCSR
import jax.numpy as jnp
import math
import numpy as np
from python_files.custom_identity import custom_identity
from stl import mesh
import scipy as sp

def stl_from_connect_and_coord(connect,coord):
  """
  connect: (nelem,3) array of element connectivity\\
  coord: (nnode,3) array of node coordinates
  """
  nelem=connect.shape[0]
  mesh_data=mesh.Mesh(np.zeros(nelem,dtype=mesh.Mesh.dtype))
  for i,f in enumerate(connect):
    for j in range(3):
      mesh_data.vectors[i][j]=coord[f[j],:]
  return mesh_data

class triMesh:
  def __init__(self,connect,coord,from_zero=True):
    if from_zero:
      self.connect=np.array(connect) #(num_elem,3)
    else:
      self.connect=np.array(connect)-1
    self.coord=np.array(coord) #(num_node,3)
    self.vectors=coord[self.connect.flatten()].reshape(-1,3,3) #(num_elem,3,3)
    self.points=self.vectors.reshape(-1,9) #(num_elem,9)
    v1=self.vectors[:,1]-self.vectors[:,0]
    v2=self.vectors[:,2]-self.vectors[:,0]
    self.normals=np.cross(v1,v2)
  
  def norm(self,scale=1.0):
    pmin=self.coord.min(axis=0)
    self.vectors=((self.vectors-pmin)*scale)-np.array([0.,1e-6,0.])
    self.coord=((self.coord-pmin)*scale)
    self.points=self.vectors.reshape(-1,9) #(num_elem,9)
    v1=self.vectors[:,1]-self.vectors[:,0]
    v2=self.vectors[:,2]-self.vectors[:,0]
    self.normals=np.cross(v1,v2)
    self.scale=scale
    self.offset=pmin
  
  def transform(self,v):
    out=(v-self.offset)*self.scale
    return out
    
def _cross_i(v1,v2,i):
  i1=(i+1)%3; i2=(i+2)%3
  return v1[:,:,i1]*v2[:,:,i2]-v1[:,:,i2]*v2[:,:,i1]

class Build_mesh:
  def __init__(self,trimesh:triMesh,length_lattice):
    points=jnp.asarray(trimesh.points.reshape(-1,3))#-jnp.array([0.,1e-6,0.])
    pmax=points.max(axis=0)
    pmin=points.min(axis=0)
    Lx,Ly,Lz=pmax-pmin
    lx,ly,lz=length_lattice
    self.nx=int(Lx/lx)+1
    self.ny=int(Ly/ly)+1
    self.nz=int(Lz/lz)+1
    self.lx,self.ly,self.lz=lx,ly,lz
    self.rx=jnp.array([lx,ly,lz])*trimesh.scale
    stls=points.reshape(-1,3,3)
    self.stl_vectors=trimesh.vectors
    self.normals=jnp.asarray(trimesh.normals) #(num_stls,3)
    self.normals=self.normals/jnp.linalg.norm(self.normals,axis=1,keepdims=True) #(num_estls,3)
    self.const=(self.normals*stls[:,0]).sum(axis=1) #(num_stls,)
    self.interceptx=self.const/self.normals[:,0] #(num_stls,)
    self.intercepty=self.const/self.normals[:,1] #(num_stls,)
    self.interceptz=self.const/self.normals[:,2] #(num_stls,)
    self.intercepts=jnp.array([self.const/self.normals[:,0],
                     self.const/self.normals[:,1],
                     self.const/self.normals[:,2]])
    self.isinfx=jnp.isinf(self.interceptx)
    self.isinfy=jnp.isinf(self.intercepty)
    self.isinfz=jnp.isinf(self.interceptz)
    self.isinfs=jnp.array([jnp.isinf(self.interceptx),jnp.isinf(self.intercepty),jnp.isinf(self.interceptz)])
    self.stl1=stls.at[:,:,0].set(0.) #(num_stls,3,3)
    self.stl2=stls.at[:,:,1].set(0.) #(num_stls,3,3)
    self.stl3=stls.at[:,:,2].set(0.) #(num_stls,3,3)
    self.stls=stls
    self.trimesh=trimesh
    self.FLAG_EDGE=True
    self.set_funcs()
    edges=trimesh.connect[:,[0,1,1,2,2,0]].reshape(-1,2)
    edges=np.sort(edges,axis=1)
    edges=np.unique(edges,axis=0)
    self.edges=trimesh.coord[edges] #(num_edges,2,3)
    self.length_lattice=np.array(length_lattice)
    
  def isin_preprocess(self,nodes,max_elem=2**35):
    """
    nodes : (num_nodes,3)
    """
    num_v_sep=max_elem//nodes.shape[0]
    num_iter=int(np.ceil(nodes.shape[0]/num_v_sep))
    seps=np.arange(0,num_iter+1)*num_v_sep
    @jit
    def _func(bnodes):
      const2=(self.normals*bnodes[:,None]).sum(axis=-1) #(num_nodes,num_stls)
      intercept2=const2/self.normals[:,2] #(num_nodes,num_stls)
      isinf=jnp.isinf(self.interceptz) #(num_stls,)
      isover=(intercept2>self.interceptz) #(num_nodes,num_stls)
      isover=jnp.logical_and(jnp.logical_not(isinf),isover) #(num_nodes,num_stls)
      node=bnodes.at[:,2].set(0.) #(num_nodes,3)
      vec=node[:,None,None,:]-self.stl3 #(num_nodes,num_stls,3,3)
      v1=jnp.sign(_cross_i(vec[:,:,0],vec[:,:,1],2)) #(num_nodes,num_stls,)
      v2=jnp.sign(_cross_i(vec[:,:,1],vec[:,:,2],2)) #(num_nodes,num_stls,)
      v3=jnp.sign(_cross_i(vec[:,:,2],vec[:,:,0],2)) #(num_nodes,num_stls,)
      is_on_edge=(v1==0)+(v2==0)+(v3==0) #(num_nodes,num_stls,)
      isintri=(jnp.abs(v1+v2+v3)==3.)+is_on_edge #(num_nodes,num_stls,)
      ispenetrate=jnp.logical_and(isintri,isover) #(num_nodes,num_stls,)
      num_penetrate=ispenetrate.sum(axis=1) #(num_nodes)
      isin=((num_penetrate%2).astype(bool)) #(num_nodes)
      return isin
    isinpoly=jnp.zeros(nodes.shape[0],dtype=bool)
    for i in range(num_iter):
      bnodes=nodes[seps[i]:seps[i+1]]
      isinpoly=isinpoly.at[seps[i]:seps[i+1]].set(_func(bnodes))
    return isinpoly

  def set_funcs(self):

    #@partial(jit,static_argnums=(1,))
    def isIn_2(nodes,nbatch=2):
      """
      nodes:(num_nodes,3)
      stls:(num_stls,3,3)
      self.normals:(num_stls,3)
      """
      n_nodes=nodes.shape[0]
      sbatch=n_nodes//nbatch
      slices=[]
      for i in range(nbatch):
        slices.append(i*sbatch)
      slices.append(n_nodes-1)
      isinpoly=[]
      for i in range(nbatch):
        bnodes=nodes[slices[i]:slices[i+1]]
        const2=(self.normals*bnodes[:,None]).sum(axis=-1) #(num_nodes,num_stls)
        intercept2=const2/self.normals[:,2] #(num_nodes,num_stls)
        isinf=jnp.isinf(self.interceptz) #(num_stls,)
        isover=(intercept2>self.interceptz) #(num_nodes,num_stls)
        isover=jnp.logical_and(jnp.logical_not(isinf),isover) #(num_nodes,num_stls)
        node=bnodes.at[:,2].set(0.) #(num_nodes,3)
        vec=node[:,None,None,:]-self.stl3 #(num_nodes,num_stls,3,3)
        v1=jnp.sign(_cross_i(vec[:,:,0],vec[:,:,1],2)) #(num_nodes,num_stls,)
        v2=jnp.sign(_cross_i(vec[:,:,1],vec[:,:,2],2)) #(num_nodes,num_stls,)
        v3=jnp.sign(_cross_i(vec[:,:,2],vec[:,:,0],2)) #(num_nodes,num_stls,)
        is_on_edge=(v1==0)+(v2==0)+(v3==0) #(num_nodes,num_stls,)
        isintri=(jnp.abs(v1+v2+v3)==3.)+is_on_edge #(num_nodes,num_stls,)
        ispenetrate=jnp.logical_and(isintri,isover) #(num_nodes,num_stls,)
        num_penetrate=ispenetrate.sum(axis=1) #(num_nodes)
        isinpoly.append((num_penetrate%2).astype(bool)) #(num_nodes)
      isinpoly=jnp.concatenate(isinpoly)
      return isinpoly
    self.isIn_2=isIn_2

    #@jit
    def vid(vi,vj,vk):
      """
      vi,vj,vk: node id coordinate (num_vid,)
      """
      out=vi+(self.nx+1)*vj+(self.nx+1)*(self.ny+1)*vk
      return out
    self.vid=vid
    
    #@jit
    def connect(ei,ej,ek):
      """
      ei,ej,ek: element id coordinate (num_elem,)
      connectivity:(num_elem,8)
      eligibility:(num_elem,)
      remeshing:(num_elem,)
      node_bools:(num_elem,8)
      """
      v1=self.vid(ei,ej,ek); v2=self.vid(ei+1,ej,ek)
      v3=self.vid(ei+1,ej+1,ek); v4=self.vid(ei,ej+1,ek)
      v5=self.vid(ei,ej,ek+1); v6=self.vid(ei+1,ej,ek+1)
      v7=self.vid(ei+1,ej+1,ek+1); v8=self.vid(ei,ej+1,ek+1)
      isin1=self.isIn[v1]; isin2=self.isIn[v2]
      isin3=self.isIn[v3]; isin4=self.isIn[v4]
      isin5=self.isIn[v5]; isin6=self.isIn[v6]
      isin7=self.isIn[v7]; isin8=self.isIn[v8]
      connectivity=jnp.array([v1,v2,v3,v4,v5,v6,v7,v8]).T
      remeshing=isin1*isin2*isin3*isin4*isin5*isin6*isin7*isin8
      return connectivity,remeshing
    self.connect=connect

  def isinpoly(self,nodes_shaped):
    """
    nodes_shaped : float (nz,ny,nx,3)
    """
    ys=nodes_shaped[0,:,0,1] #(ny,)
    x_min,x_max=interception_x(self.edges,ys) #(ny,)
    msk=(nodes_shaped[:,:,:,0]>=x_min[:,None])*(nodes_shaped[:,:,:,0]<=x_max[:,None]) #(nz,ny,nx)
    msk_flatten=msk.flatten()
    nodes=nodes_shaped.reshape(-1,3)
    #msk_flatten=msk_flatten.at[0].set(True)
    msk_isinpoly=msk_flatten.at[jnp.where(msk_flatten)[0]].set(self.isin_preprocess(nodes[msk_flatten])) #(nz*ny*nx,)
    return msk_isinpoly,nodes
    
  def _fill_mesh(self,nbatch=2):
    
    gx=jnp.linspace(0.,self.nx*self.lx,self.nx+1)
    gy=jnp.linspace(0.,self.ny*self.ly,self.ny+1)
    gz=jnp.linspace(0.,self.nz*self.lz,self.nz+1)
    gx,gy,gz=jnp.meshgrid(gx,gy,gz)
    gx,gy,gz=map(lambda x:x.transpose(2,0,1).flatten(),[gx,gy,gz])
    self.nodes=jnp.array([gx,gy,gz]).T #(num_nodes,3)
    self.nodes_shaped=self.nodes.reshape(self.nz+1,self.ny+1,self.nx+1,3)
    #self.isIn=self.isIn_2(self.nodes,nbatch)
    self.isIn,self.nodes=self.isinpoly(self.nodes_shaped)
    ei=jnp.arange(self.nx); ej=jnp.arange(self.ny); ek=jnp.arange(self.nz)
    ei,ej,ek=jnp.meshgrid(ei,ej,ek)
    ei,ej,ek=map(lambda x:x.transpose(2,0,1).flatten(),[ei,ej,ek])
    self.connectivity,self.remeshing=self.connect(ei,ej,ek)
    
  def _delete_unused_nodes(self):
    used_nid,self.connectivity=jnp.unique(self.connectivity,return_inverse=True)
    self.coordinates=self.coordinates[used_nid]
    
  def remesh_element(self,coord_skin=None,connect_skin=None,scale=None,offset=None):
    """
    connectivity:(num_elem,8)
    """
    if coord_skin is None:
      coord_skin=self.trimesh.coord
      connect_skin=self.trimesh.connect
      scale=self.trimesh.scale
      offset=self.trimesh.offset
    self._fill_mesh()
    self.connectivity=self.connectivity[jnp.where(self.remeshing)[0]]
    self.coordinates=self.nodes
    
    indice=jnp.array([[0,1,2,3],[4,5,6,7],[0,4,5,1],[3,7,6,2],[0,3,7,4],[1,2,6,5]])
    self._delete_unused_nodes()
    self.connectivity,self.coordinates,self.nid_const=eliminate_invalid_grid(self.connectivity,self.coordinates)
  
  def get_static_nid(self):
    u_nodes,counts=jnp.unique(self.connectivity,return_counts=True)
    variable_nid1=jnp.where(counts==6)[0]
    variable_nid2=jnp.where((self.coordinates[:,1]==0.0)*(counts==5))[0]
    static_nid=jnp.setdiff1d(jnp.arange(self.coordinates.shape[0]),jnp.concatenate([variable_nid1,variable_nid2]))
    return static_nid

def extract_root_edge(connect,coord):
  """
  connect : (ne,3)
  coord : (nv,3)
  """
  if coord.shape[0]==0:
    return jnp.zeros((0,2),dtype=int)
  elif coord[:,1].min()>0.0:
    return jnp.zeros((0,2),dtype=int)
  root_nid=jnp.where(coord[:,1]==0.0)[0] #(nr,)
  msk_is_root_in_tri=jnp.isin(connect,root_nid) #(ne,3)
  two_root_in_tri=(msk_is_root_in_tri.sum(axis=1)==2) #(ne,)
  i1,i2=jnp.where(msk_is_root_in_tri[two_root_in_tri])
  segment=connect[two_root_in_tri][i1,i2].reshape(-1,2) #(ns,2)
  return segment

def mesh3d_to_coord_and_connect(mesh3d,round_f=4):
  """
  mesh3d : float (n_v,3,3)
  """
  _coords=mesh3d.reshape(-1,3)
  _connects=jnp.arange(_coords.shape[0]).reshape(-1,3)
  _,coords_index,inverse=jnp.unique(jnp.round(_coords,round_f),axis=0,return_index=True,return_inverse=True,)
  connects=inverse[_connects]
  return coords_index,connects

def reconstruct_coordinates(tri_coord_in,coord_static,nid_variable,surface_mapping_in):
  """
  tri_coord_in : float (n_v_tri_variable,3)\\
  coord_static : float (n_v,3)\\
  nid_variable : int (n_v_variable,)\\
  surface_mapping_in : BCOO (n_v_variable,n_v_tri_variable)\\
  
  Return
  ------
  coord_variable : float (n_v,3)
  """
  coord_variable=surface_mapping_in@tri_coord_in #(n_v_variable,3)
  v2=coord_static[nid_variable]
  v1=coord_variable[nid_variable]
  v_insert=custom_identity(v1,v2)
  coord_merged=coord_static.at[nid_variable].set(v_insert)
  return coord_merged

def count_elements(a,m):
  """
  a : (n,) array of integers
  m : int
  """
  unique_val,counts=jnp.unique(a,return_counts=True)
  out=jnp.zeros(m,dtype=int).at[unique_val].set(counts)
  return out

def cs_rbf(v,dmul):
  """
  Compactly Supported Radial Basis Function\n
  v : (n,,3)
  dmul : float
  """
  dif=v[:,None]-v #(n,n,3)
  dist=jnp.linalg.norm(dif,axis=-1) #(n,n)
  id1,id2=jnp.where(dist<dmul) #(n2,)
  r=dist[id1,id2]/dmul #(n2,)
  weight=jnp.maximum(0,1-r)**4*(4*r+1) #(n2,)
  weight=BCOO((weight,jnp.array([id1,id2]).T),shape=(v.shape[0],v.shape[0])) #(n,n)
  return weight

def connect2adjoint(connect):
  """
  Calculate adjoint matrix from node connectivity matrix in BCOO format\n
  connect : (n,8) array of integers
  """
  n=connect.max()+1
  ind=jnp.array([[0,1],[0,3],[0,4],[1,2],[1,5],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]) #(12,2)
  ind1,ind2=ind.T #(2,12)
  ind1=connect[:,ind1].flatten() #(12n,)
  ind2=connect[:,ind2].flatten() #(12n,)
  ind3=jnp.arange(n) #(n,)
  indices=jnp.array([jnp.concatenate((ind1,ind2,ind3)),jnp.concatenate((ind2,ind1,ind3))]).T #(24n,2)
  adjoint=BCOO((jnp.ones(indices.shape[0],bool),indices),shape=(n,n)) #(n,n)
  adjoint=convertBCOO2BCSR(adjoint)
  adjoint=sp.sparse.csr_matrix((adjoint.data,adjoint.indices,adjoint.indptr),shape=adjoint.shape)
  return adjoint

def cs_rbf_adjoint(v,dmul,adjoint):
  """
  Compactly Supported Radial Basis Function\n
  v : (n,,3)
  dmul : float
  """
  n=math.ceil(dmul*jnp.sqrt(3))-1
  ad=adjoint@adjoint
  for _ in range(n):
    ad=ad@adjoint
  ad_coo=ad.tocoo()
  i=ad_coo.row; j=ad_coo.col
  dif=v[i]-v[j] #(n2,3)
  dist=jnp.linalg.norm(dif,axis=-1) #(n2,)
  r=dist/dmul #(n2,)
  weight=jnp.maximum(0,1-r)**4*(4*r+1) #(n2,)
  msk=jnp.where(weight!=0.0)[0]
  data=weight[msk]
  indices=jnp.array([i[msk],j[msk]]).T
  weight=BCOO((data,indices),shape=ad_coo.shape) #(n,n)
  norm=weight@jnp.ones(weight.shape[1]) #(n,)
  assert norm.min()>0.0
  weight=convertBCOO2BCSR(weight/norm[:,None])
  return weight

def interception_x(edges,ys):
  """
  edges : float (ne,2,3)
  ys : float (ny,)
  """
  msk1=(edges[:,0,1]<=ys[:,None])*(edges[:,1,1]>=ys[:,None]) #(ny,ne)
  msk2=(edges[:,0,1]>=ys[:,None])*(edges[:,1,1]<=ys[:,None]) #(ny,ne)
  msk=msk1+msk2
  xs=(edges[:,0,0]*(edges[:,1,1]-ys[:,None])+edges[:,1,0]*(ys[:,None]-edges[:,0,1]))/(edges[:,1,1]-edges[:,0,1]) #(ny,ne)
  xs=xs.at[~msk].set(jnp.nan)
  x_min=jnp.nanmin(xs,axis=1) #(ny,)
  x_max=jnp.nanmax(xs,axis=1) #(ny,)
  return x_min,x_max

def convertBCOO2BCSR(bcoo):
  data=bcoo.data; indices=bcoo.indices; shape=bcoo.shape
  coo=sp.sparse.coo_array((data,(indices[:,0],indices[:,1])),shape=shape)
  csr=coo.tocsr()
  return BCSR.from_scipy_sparse(csr)

def eliminate_invalid_grid(connect,coord):
  """
  connect : (ne,8)
  coord : (nn,3)
  nid_const : (nc,)
  """
  
  _,counts=jnp.unique(connect,return_counts=True)
  msk_root_nid=(coord[:,1]==coord[:,1].min())
  msk_under_8=counts<8
  msk_under_4=counts<4
  msk_const_node=(msk_under_8*(~msk_root_nid)+msk_under_4*msk_root_nid)
  nid_const=jnp.where(msk_const_node)[0]

  msk_variable=jnp.ones(coord.shape[0],dtype=bool).at[nid_const].set(False) #(nn,)
  msk_valid=jnp.any(msk_variable[connect],axis=1) #(ne,)
  _connect=connect[msk_valid]
  nid_valid,connect_new=jnp.unique(_connect,return_inverse=True,)
  coord_new=coord[nid_valid]
  
  _,counts=jnp.unique(connect_new,return_counts=True)
  msk_root_nid=(coord_new[:,1]==coord_new[:,1].min())
  msk_under_8=counts<8
  msk_under_4=counts<4
  msk_const_node=(msk_under_8*(~msk_root_nid)+msk_under_4*msk_root_nid)
  nid_const=jnp.where(msk_const_node)[0]
  return connect_new,coord_new,nid_const