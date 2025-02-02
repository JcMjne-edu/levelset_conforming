import numpy as np
import scipy as sp
from lsto.mesh_tools import mapping_dist
from jax.experimental.sparse import BCSR
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

THREASHOLD=1e-10

def mapping_surfacenode_full(connects_ls,coords_ls,coords_geom,elems_tet,nodes_tet,faces_tet):
  """
  Calculate mapping from surface nodes of the levelset to the surface nodes of the fem mesh

  connects_ls : (ne,3)
  coords_ls : (nc,3)
  connects_geom : (ne_s,3)
  coords_geom : (nc_s,3)
  nodes_tet : (nn,3)
  elems_tet : (ne,4)
  """
  #print('1')
  #nid_surf_tet,edge_surf_tet=_get_nid_surf(elems_tet)
  nid_surf_tet=np.unique(faces_tet)
  edge_surf_tet=faces_tet[:,[0,1,1,2,2,0]].reshape(-1,2)
  edge_surf_tet=np.unique(np.sort(edge_surf_tet, axis=1), axis=0)
  #print('2')
  _nid_identical_tet2geom,_,_map_geom2tet=_get_nid_identical(nodes_tet[nid_surf_tet],coords_geom)
  #print('3')
  map_geom2tet=_expand_mat_row(_map_geom2tet,nid_surf_tet,nodes_tet.shape[0])
  #print('4')
  nid_identical_tet2geom=nid_surf_tet[_nid_identical_tet2geom]
  nid_additional_tet=np.setdiff1d(nid_surf_tet,nid_identical_tet2geom)
  _,_,map_ls2geom=_get_nid_identical(coords_geom,coords_ls)
  #print('5')
  adj_tet_full=_get_adjecent_mat(edge_surf_tet,nodes_tet.shape[0])
  #print('6')
  adj_root2additional=_find_nearest_root(adj_tet_full,nid_additional_tet,nid_identical_tet2geom,3)
  #print('7')
  table_trils=_get_table_nid2tri(connects_ls,coords_ls.shape[0])
  #print('8')
  mat_identical=map_geom2tet@map_ls2geom
  table_additional2trils=adj_root2additional@mat_identical@table_trils
  mat_weight_additional,nid_valid_tet=_get_mat_weight(table_additional2trils,nodes_tet,connects_ls,coords_ls)
  mat_weight_identical=mat_identical.astype(float)
  mat_weight,nid_valid_tet=_combine_weight(mat_weight_identical,mat_weight_additional)
  mat_weight=BCSR.from_scipy_sparse(mat_weight)
  return mat_weight,nid_valid_tet,nid_surf_tet
  
def _get_nid_surf(elems_tet):
  """
  Calculate the surface node id of the tetrahedral mesh.

  elems_tet : (ne,4)
  """
  faces=elems_tet[:,[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]].reshape(-1,3)
  faces=np.sort(faces, axis=1)
  u_faces,counts=np.unique(faces, return_counts=True, axis=0)
  face_surf=u_faces[counts==1]
  nid_surf=np.unique(face_surf)
  edge_surf=face_surf[:,[0,1,1,2,2,0]].reshape(-1,2)
  edge_surf=np.unique(np.sort(edge_surf, axis=1), axis=0)
  return nid_surf,edge_surf

def _get_nid_identical(coords1,coords2):
  """
  Calculate the identical node ids between two different node systems
  Ideally, coords1[nid_identical1] == coords2[nid_identical2]

  coords1 : (n1,3)
  coords2 : (n2,3)
  mapping : (n1,n2)

  """
  n1,n2=coords1.shape[0],coords2.shape[0]
  _nid_identical_2,min_dist=mapping_dist.calc_dist(coords1,coords2)
  nid_identical_1=np.where(min_dist<THREASHOLD)[0] #(m,)
  nid_identical_2=_nid_identical_2[nid_identical_1] #(m,)
  data=np.ones(nid_identical_1.shape[0],bool)
  mapping=sp.sparse.csc_array((data,(nid_identical_1,nid_identical_2)),shape=(n1,n2))
  return np.array(nid_identical_1), np.array(nid_identical_2), mapping

def _get_adjecent_mat(edge,size):
  """
  Calculate the adjecent matrix of the mesh

  edge : (ne,2)
  """
  ix=edge[:,0]
  iy=edge[:,1]
  data=np.ones(ix.shape[0],bool)
  csc_mat=sp.sparse.csc_array((data,(ix,iy)),shape=(size,size))
  ix_diag=np.arange(size);iy_diag=np.arange(size)
  data_diag=np.zeros(size,bool)
  csc_mat_diag=sp.sparse.csc_array((data_diag,(ix_diag,iy_diag)),shape=(size,size))
  
  adjecent_mat=(csc_mat+csc_mat.T+csc_mat_diag)
  return adjecent_mat

def _find_nearest_root(adj,nid_trg,nid_root,num_nearest):
  """
  Find the nearest node of the target node from the root node

  adj : (n,n)
  nid_trg : (m,)
  nid_root : (l,)
  num_nearest : int
  """
  n=adj.shape[0]
  adj_root2additional=adj[:,nid_root] #(n,l)
  num_root=_get_metric_adj(adj_root2additional,nid_trg) #(m,)
  while num_root.min()<num_nearest:
    _adj_root2additional=adj@adj_root2additional
    idx_invalid=nid_trg[num_root<num_nearest]
    idx_valid=np.setdiff1d(np.arange(n),idx_invalid)
    msk_valid=sp.sparse.csr_array((np.ones(idx_valid.shape[0],bool),(idx_valid,idx_valid)),shape=adj.shape)
    mask_invalid=sp.sparse.csr_array((np.ones(idx_invalid.shape[0],bool),(idx_invalid,idx_invalid)),shape=adj.shape)
    adj_root2additional=msk_valid@adj_root2additional+mask_invalid@_adj_root2additional
    num_root=_get_metric_adj(adj_root2additional,nid_trg)
  idx1=np.arange(nid_root.shape[0])
  idx2=nid_root
  data=np.ones(nid_root.shape[0],bool)
  mat_e=sp.sparse.csc_array((data,(idx1,idx2)),shape=(nid_root.shape[0],adj.shape[0]))
  data2=np.ones(nid_trg.shape[0],bool)
  mat_s=sp.sparse.csc_array((data2,(nid_trg,nid_trg)),shape=adj.shape)
  adj_root2additional_ex=mat_s@adj_root2additional@mat_e #(n,n)
  return adj_root2additional_ex

def _get_metric_adj(mat,nid_trg):
  """
  Calculate the sum of the adjecent root nodes of the target nodes
  nid_trg : (m,)
  nid_root : (n,)
  """
  mat_reduced=mat[nid_trg] #(m,n)
  num_roots=mat_reduced.sum(axis=1) #(m,)
  return num_roots

def _get_table_nid2tri(tris,m):
  """
  tris : (n,3)
  m : int (number of nodes)
  """
  n=tris.shape[0]
  idx1=tris.flatten()
  idx2=np.arange(tris.shape[0]).repeat(3)
  data=np.ones(idx1.shape[0],bool)
  map_nid2tri=sp.sparse.csc_array((data,(idx1,idx2)),shape=(m,n))
  return map_nid2tri

def _get_invdist_if_inside(v,tri_verts):
  """
  Calculate the distance of point from triangles if the point is inside the triangle.
  v: (n,3)
  tri_verts: (n,3,3) float
  """
  tri_norm=np.cross(tri_verts[:,1]-tri_verts[:,0],tri_verts[:,2]-tri_verts[:,0]) #(n,3)
  tri_norm_normed=tri_norm/np.linalg.norm(tri_norm,axis=1,keepdims=True) #(n,3)
  vs=tri_verts-v[:,None,:] #(n,3,3)
  v_norm_dot1,v_norm_dot2,v_norm_dot3=_norm_dot(vs,tri_norm_normed) #(n,)
  inside_tri_p=(v_norm_dot1>-THREASHOLD)*(v_norm_dot2>-THREASHOLD)*(v_norm_dot3>-THREASHOLD) #(n,)
  inside_tri_n=(v_norm_dot1<THREASHOLD)*(v_norm_dot2<THREASHOLD)*(v_norm_dot3<THREASHOLD) #(n,)
  inside_tri=inside_tri_p+inside_tri_n #(n,)
  dist=np.abs((vs[:,0]*tri_norm_normed).sum(axis=1)) #(n,)
  invdist=1/(dist+1e-6)
  invdist=invdist*inside_tri*(dist<1e-6)
  return invdist

def _norm_dot(vs,tri_norm_normed):
  """
  v1 : float (n,3)
  v2 : float (n,3)
  n : float (n,3)
  """
  v_norm01=np.cross(vs[:,0],vs[:,1]) #(n,3)
  v_norm12=np.cross(vs[:,1],vs[:,2]) #(n,3)
  v_norm20=np.cross(vs[:,2],vs[:,0]) #(n,3)
  v_norm_dot1=(v_norm01*tri_norm_normed).sum(axis=1) #(n,)
  v_norm_dot2=(v_norm12*tri_norm_normed).sum(axis=1)
  v_norm_dot3=(v_norm20*tri_norm_normed).sum(axis=1)
  return v_norm_dot1,v_norm_dot2,v_norm_dot3

def _get_mat_weight(table,coords1,connects2,coords2):
  """
  Calculate the weight of the mapping from coords2 to coords1

  table : csr_array (m,n)
  coords1 : (n1,3)
  connects2 : (ne,3)
  coords2 : (n2,3)
  """
  table_coo=table.tocoo() #(m,n)
  nid1=table_coo.row #(nnz,)
  nid2=table_coo.col #(nnz,)
  v=coords1[nid1]
  tri_verts=coords2[connects2[nid2]] #(nnz,3,3)
  invdist=_get_invdist_if_inside(v,tri_verts) #(nnz,)
  table_dist=sp.sparse.csr_array((invdist,(nid1,nid2)),shape=table.shape) #(m,n)
  nid1_valid=table_dist.sum(axis=1).nonzero()[0] #(m1,)
  tri_idx=table_dist[nid1_valid].argmax(axis=1)[:,0] #(m1,)
  v_valid=coords1[nid1_valid] #(m1,3)

  #calculate the weight
  tri_coord=coords2[connects2[tri_idx]] #(m1,3,3)
  b_a=tri_coord[:,1]-tri_coord[:,0] #(m1,3)
  c_a=tri_coord[:,2]-tri_coord[:,0] #(m1,3)
  v_a=v_valid-tri_coord[:,0] #(m1,3)
  coeff=np.linalg.pinv(np.array([b_a,c_a]).transpose(1,0,2)) #(m1,3,2)
  coeff=(v_a[:,:,None]*coeff).sum(axis=1) #(m1,2)
  weight=np.concatenate([1.-coeff.sum(axis=1,keepdims=True),coeff],axis=1) #(m1,3)
  idx1=nid1_valid.repeat(3) #(3m1,)
  idx2=connects2[tri_idx].flatten() #(3m1,)
  data=weight.flatten() #(3m1,)
  mat=sp.sparse.csr_array((data,(idx1,idx2)),shape=(coords1.shape[0],coords2.shape[0])) #(n1,n2)
  return mat,nid1_valid

def _combine_weight(mat1,mat2):
  """
  Combine two weight matrices
  """
  mat_combined=mat1+mat2
  row_valid=mat_combined.sum(axis=1).nonzero()[0]
  mat_combined=mat_combined[row_valid]
  return mat_combined,row_valid

def _expand_mat_row(mat,nodes,n):
  idx=np.arange(nodes.shape[0])
  data=np.ones(nodes.shape[0],bool)
  mat_e=sp.sparse.csr_array((data,(idx,idx)),shape=(n,nodes.shape[0]))
  return mat_e@mat