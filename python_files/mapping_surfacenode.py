import numpy as np
import scipy as sp
from jax import jit
import jax.numpy as jnp
from jax.experimental.sparse import BCSR

def mapping_surfacenode(coords_ls,connects_ls,nodes_tet,elems_tet,nid_identical,nid_inside):
  n_coord_ls=coords_ls.shape[0]
  print(coords_ls.shape,connects_ls.shape)
  indice1=np.array([nid_identical,np.arange(n_coord_ls)]).T #(n_v1,2)
  data1=np.ones(n_coord_ls) #(n_v1,)
  mat1=sp.sparse.csr_matrix((data1,(indice1[:,0],indice1[:,1])),shape=(nodes_tet.shape[0],n_coord_ls))
  nid_additional=np.setdiff1d(nid_inside,nid_identical)
  
  metric_msk=_get_metric_msk(nid_additional,nid_identical,nid_inside,elems_tet,nodes_tet,connects_ls)
  table_nid=_get_table_nid(metric_msk)
  mapping=_get_map_node2tri(connects_ls,table_nid)
  inside_tri,dist=_isinside_dist(mapping.row,mapping.col,connects_ls,coords_ls,nodes_tet[nid_additional])
  table_dist_inv=_get_table_dist(inside_tri,dist,mapping.row,mapping.col,mapping.shape)
  eid_trg=table_dist_inv.argmax(axis=0)
  weight=_get_weight(connects_ls,coords_ls,eid_trg,nodes_tet[nid_additional])
  
  ind1=nid_additional.repeat(3)
  ind2=np.array(connects_ls[eid_trg].flatten()) #(n_v2*3,)
  indice2=np.stack([ind1,ind2],axis=1) #(n,2)
  data2=np.asarray(weight.flatten()) #(n,)
  mat2=sp.sparse.csr_matrix((data2,(indice2[:,0],indice2[:,1])),shape=mat1.shape)
  surface_mapping=(mat1+mat2)[nid_inside]
  surface_mapping=BCSR.from_scipy_sparse(surface_mapping)
  return surface_mapping,(table_nid,nid_additional,table_dist_inv,inside_tri,mapping)

def _isinside_dist(eid,nid,connects_ls,coords_ls,nodes):
  """
  eid : (n)
  nid : (n,)
  connects_ls : (ne,3)
  coords_ls : (nc,3)
  nodes : (nn,3)

  Check if the node is inside the triangle and calculate the distance
  """
  THREASHOLD=1e-12
  tri_stl=coords_ls[connects_ls] #(ne,3,3)
  tri_stl_norm=jnp.cross(tri_stl[:,1]-tri_stl[:,0],tri_stl[:,2]-tri_stl[:,0]) #(ne,3)
  tri_stl_norm_normed=tri_stl_norm/jnp.linalg.norm(tri_stl_norm,axis=1,keepdims=True) #(ne,3)
  tri_stl_norm_normed=tri_stl_norm_normed[eid] #(n,3)
  vs=tri_stl[eid]-nodes[nid,None,:] #(n,3,3)
  v_norm_dot1,v_norm_dot2,v_norm_dot3=_norm_dot(vs,tri_stl_norm_normed) #(n,)
  inside_tri_p=(v_norm_dot1>-THREASHOLD)*(v_norm_dot2>-THREASHOLD)*(v_norm_dot3>-THREASHOLD) #(n,)
  inside_tri_n=(v_norm_dot1<THREASHOLD)*(v_norm_dot2<THREASHOLD)*(v_norm_dot3<THREASHOLD) #(n,)
  inside_tri=inside_tri_p+inside_tri_n #(n,)
  dist=np.abs((vs[:,0]*tri_stl_norm_normed).sum(axis=1)) #(n,)
  return inside_tri,dist

@jit
def _norm_dot(vs,tri_norm_normed):
  """
  v1 : float (n,3)
  v2 : float (n,3)
  n : float (n,3)
  """
  v_norm01=jnp.cross(vs[:,0],vs[:,1]) #(n,3)
  v_norm12=jnp.cross(vs[:,1],vs[:,2]) #(n,3)
  v_norm20=jnp.cross(vs[:,2],vs[:,0]) #(n,3)
  v_norm_dot1=(v_norm01*tri_norm_normed).sum(axis=1) #(n,)
  v_norm_dot2=(v_norm12*tri_norm_normed).sum(axis=1)
  v_norm_dot3=(v_norm20*tri_norm_normed).sum(axis=1)
  return v_norm_dot1,v_norm_dot2,v_norm_dot3

def _get_map_node2tri(connects_ls,table_nid):
  ix=np.arange(connects_ls.shape[0]).repeat(3)
  iy=connects_ls.flatten()
  data=np.ones(ix.shape[0],bool)
  csc_mat=sp.sparse.csc_array((data,(ix,iy)),shape=(ix.max()+1,iy.max()+2))
  mat_full=csc_mat[:,table_nid.flatten()] 

  ix2=np.arange(table_nid.shape[0]*table_nid.shape[1])
  iy2=np.arange(table_nid.shape[0]).repeat(table_nid.shape[1])
  data2=np.ones(iy2.shape[0],bool)
  csc_mat2=sp.sparse.csc_matrix((data2,(ix2,iy2)),shape=(ix2.max()+1,iy2.max()+1))
  
  mapping=(mat_full@csc_mat2).tocoo()
  return mapping

def _get_metric_msk(nid_additional,nid_identical,nid_inside,elems_tet,nodes_tet,connects_ls):
  faces=elems_tet[:,np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])].reshape(-1,3) #(nf,3)
  u_faces,counts=np.unique(np.sort(faces,axis=1),return_counts=True,axis=0) #(nf,3)
  face_surf=u_faces[counts==1] #(nf1,3)
  edge_surf=face_surf[:,np.array([[0,1],[0,2],[1,2]])].reshape(-1,2) #(nf1*3,2)
  edge_ls=nid_identical[connects_ls][:,np.array([[0,1],[0,2],[1,2]])].reshape(-1,2)
  edge_surf=np.concatenate([edge_surf,edge_ls])
  edge_surf=np.unique(np.sort(edge_surf,axis=1),axis=0) #(ne,2)
  eid_prime=np.isin(edge_surf,nid_identical).any(axis=1)
  edge_prime=edge_surf[eid_prime]
  diag=nid_inside.repeat(2).reshape(-1,2)
  _indices=np.concatenate([diag,edge_prime,edge_prime[:,::-1]])
  _data=np.ones(_indices.shape[0],bool)
  csr_mat_prime=sp.sparse.csr_array((_data,(_indices[:,0],_indices[:,1])),shape=(nodes_tet.shape[0],nodes_tet.shape[0]))
  #calculate adjacency matrix
  diag=np.arange(nodes_tet.shape[0]).repeat(2).reshape(-1,2)
  _indices=np.concatenate([diag,edge_surf,edge_surf[:,::-1]])
  _data=np.ones(_indices.shape[0],bool)
  csr_mat_full=sp.sparse.csr_array((_data,(_indices[:,0],_indices[:,1])),shape=(nodes_tet.shape[0],nodes_tet.shape[0]))

  label=np.zeros(nid_additional.shape[0],bool)
  temp=csr_mat_prime
  metric_msk=sp.sparse.csr_array((nid_additional.shape[0],nid_identical.shape[0]))
  metric=temp[nid_additional][:,nid_identical]
  msk_i=np.where((metric.sum(axis=1)>3)*(~label))[0]
  label[msk_i]=True
  msk_data=np.ones(msk_i.shape[0],bool)
  msk=sp.sparse.csr_array((msk_data,(msk_i,msk_i)),shape=(label.shape[0],label.shape[0]))
  metric_msk=msk@metric
  n=5
  while True:
    _temp=csr_mat_full@temp
    metric=_temp[nid_additional][:,nid_identical]
    msk_i=np.where((metric.sum(axis=1)>=n)*(~label))[0]
    label[msk_i]=True
    msk_data=np.ones(msk_i.shape[0],bool)
    msk=sp.sparse.csr_array((msk_data,(msk_i,msk_i)),shape=(label.shape[0],label.shape[0]))
    metric_msk+=msk@metric
    if metric.sum(axis=1).min()>=n:
      break
    temp=_temp
  return metric_msk

def _get_table_dist(inside_tri,dist,eid,nid,shape):
  dist[~inside_tri]+=1e5
  dist=1./(dist+1e-6)
  table_dist=sp.sparse.csc_array((dist,(eid,nid)),shape=shape)
  return table_dist

def _get_table_nid(metric):
  """
  calculate the table of surrounding key node ids of each target node
  indeces are of coords_ls
  """
  idx1=np.zeros(metric.indices.shape[0],int)
  idx2=np.zeros(metric.indices.shape[0],int)
  for i,p1 in enumerate(metric.indptr[:-1]):
    p2=metric.indptr[i+1]
    idx1[p1:p2]=i
    idx2[p1:p2]=np.arange(p2-p1)
  table_nid=-np.ones((metric.shape[0],idx2.max()+1),int)
  table_nid[idx1,idx2]=metric.indices
  return table_nid

def _get_weight(connects_ls,coords_ls,eid,nodes):
  """
  Calculate the weight of each node in the triangle
  connects_ls : (ne,3)
  coords_ls : (nc,3)
  eid : (n,)
  nodes : (n,3)
  """
  tri_coord=coords_ls[connects_ls[eid]] #(n,3,3)
  b_a=tri_coord[:,1]-tri_coord[:,0] #(n,3)
  c_a=tri_coord[:,2]-tri_coord[:,0] #(n,3)
  v_a=nodes-tri_coord[:,0] #(n,3)
  coeff=jnp.linalg.pinv(jnp.array([b_a,c_a]).transpose(1,0,2)) #(n,3,2)
  coeff=(v_a[:,:,None]*coeff).sum(axis=1) #(n,2)
  weight=jnp.concatenate([1.-coeff.sum(axis=1,keepdims=True),coeff],axis=1) #(n,3)
  return weight
