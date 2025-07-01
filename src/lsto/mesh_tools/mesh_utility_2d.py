import numpy as np
import scipy as sp

def make_adj2d(useg):
  """
  Compute adjacency matrix from usegment
  useg : (n,2) int
  """
  indices=np.concatenate([useg,useg[:,::-1]],axis=0)
  datas=np.ones(indices.shape[0],bool)
  nnode=useg.max()+1
  adj_mat=sp.sparse.csr_matrix((datas,(indices[:,0],indices[:,1])),shape=(nnode,nnode))
  return adj_mat

def points_in_holes2d(coord,seg):
  unid,inv=np.unique(seg,return_inverse=True)
  _coord=coord[unid]
  _seg=inv.reshape(-1,2)
  adj_mat=make_adj2d(_seg)
  n_components,label=sp.sparse.csgraph.connected_components(csgraph=adj_mat,directed=False)
  labe_seg=label[_seg[:,0]]

def point_in_hole2d(vert):
  pass