import numpy as np

def _replace_nid(nnode,edge_trg,connect):
  map_replace=np.arange(nnode)
  map_replace[edge_trg[:,1]]=edge_trg[:,0]
  return map_replace[connect]

def _area_from_coord_connect(coord,connect):
  """
  cooord : (n,3)
  connect : (m,3)
  """
  v=coord[connect] # (m,3,3)
  area=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1) # (m,)
  return area

def merge_close_nodes(connects,coords,threashold=1e-6):
  while True:
    edge=connects[:,[0,1,1,2,2,0]].reshape(-1,2)
    edge=np.unique(np.sort(edge,axis=1),axis=0)
    l=np.linalg.norm(coords[edge[:,0]]-coords[edge[:,1]],axis=1)
    msk_l=(l<threashold)
    if not msk_l.any():
      break
    edge=edge[msk_l]
    l=l[msk_l]
    sort_idx=np.argsort(l)
    edge=edge[sort_idx]
    nids_used=np.zeros(coords.shape[0],dtype=bool)
    edge_trg=np.zeros(edge.shape[0],dtype=bool)
    for i,eid in enumerate(edge_trg):
      if not nids_used[eid].any():
        edge_trg[i]=True
        nids_used[eid]=True
    edge_trg=edge[edge_trg] # (m,2)
    coord_additional=coords[edge_trg].mean(axis=1) # (m,3)
    coords[edge_trg[:,0]]=coord_additional
    connects=_replace_nid(coords.shape[0],edge_trg,connects)
    area=_area_from_coord_connect(coords,connects)
    connects=connects[area>0.0]
    u,inv=np.unique(connects,return_inverse=True)
    connects=inv.reshape(connects.shape)
    coords=coords[u]
  return connects,coords