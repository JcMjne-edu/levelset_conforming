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

def setup(connects,coords):
  """
  connects: (m,3)
  coords: (n,3)
  """
  edge=connects[:,[[0,1],[1,2],[2,0]]] #(m,3,2)
  edge=np.sort(edge,axis=2) #(m,3,2)
  l=np.linalg.norm(coords[edge][:,:,0]-coords[edge][:,:,1],axis=2) #(m,3)
  l_argmax=l.argmax(axis=1) #(m,)
  vs=coords[connects]
  area=np.linalg.norm(np.cross(vs[:,1]-vs[:,0],vs[:,2]-vs[:,0]),axis=1)#/2
  aspect=area/l.max(axis=1)**2 #(m,)
  min_max_ratio=l.min(axis=1)/l.max(axis=1)
  u_edge,inv,counts=np.unique(edge.reshape(-1,2),axis=0,return_inverse=True,return_counts=True)
  if counts.min()==1:
    raise ValueError('Holes in mesh')
  map_f2e=inv.reshape(-1,3) #(m,3)
  #mapping from edge to face
  map_e2f=np.arange(connects.shape[0]).repeat(3)[np.argsort(inv)].reshape(-1,2) #(m,2)
  eid_longest=map_f2e[np.arange(len(edge)),l_argmax] #(m,)
  return aspect,eid_longest,min_max_ratio,u_edge,map_e2f,map_f2e

def get_aspect(connects,coords,u_edge,map_f2e,fid_updated):
  """
  connects: (m,3)
  coords: (n,3)
  u_edge: (k,2)
  map_f2e: (m,3)
  fid_updated: (mu,)
  """
  mu=fid_updated.shape[0]
  eids=u_edge[map_f2e[fid_updated]] #(mu,3,2)
  v=coords[eids] #(mu,3,2,3)
  l=np.linalg.norm(v[:,:,0]-v[:,:,1],axis=2) #(mu,3)
  l_argmax=l.argmax(axis=1) #(mu,)
  vs=coords[connects[fid_updated]] #(mu,3,3)
  area=np.linalg.norm(np.cross(vs[:,1]-vs[:,0],vs[:,2]-vs[:,0]),axis=1)
  aspect=area/l.max(axis=1)**2
  min_max_ratio=l.min(axis=1)/l.max(axis=1)
  eid_longest=(map_f2e[fid_updated])[np.arange(mu),l_argmax] #(m,)
  return aspect,eid_longest,min_max_ratio

def extract_unique_eid_trg(eid_trg,u_edge,nnode):
  edge_trg=u_edge[eid_trg] #(m,2)
  nid_used=np.zeros(nnode,dtype=bool)
  unique_edge_id=np.zeros(len(eid_trg),dtype=bool)
  for i,e in enumerate(edge_trg):
    if not nid_used[e].any():
      nid_used[e]=True
      unique_edge_id[i]=True
  return eid_trg[unique_edge_id]

def eliminate_lowaspect_triangle(connects,coords,threshold=1e-3):
  new_connects=connects.copy()
  nnode=coords.shape[0]
  for i in range(20):
    aspect,eid_longest,min_max_ratio,u_edge,map_e2f,map_f2e=setup(new_connects,coords)
    if aspect.min()>threshold:
      break
    msk_trg=(aspect<threshold)*(min_max_ratio>0.1)
    if msk_trg.sum()==0:
      break
    aspect_trg=aspect[msk_trg]
    _eid_trg=eid_longest[msk_trg]
    #aspect_argsort=aspect_trg.argsort()
    aspect_argsort=np.arange(aspect_trg.shape[0])
    np.random.shuffle(aspect_argsort)
    eid_trg=_eid_trg[aspect_argsort]
    eid_selected=extract_unique_eid_trg(eid_trg,u_edge,nnode) #(ms,)
    fid_selected=map_e2f[eid_selected] #(ms,2)
    nid_selected=new_connects[fid_selected] #(ms,2,3)
    new_edge=np.zeros((eid_selected.shape[0],2),int) #(ms,2)
    for i,nids in enumerate(nid_selected):
      _,inv,count=np.unique(nids,return_inverse=True,return_counts=True)
      count[inv.reshape(2,3)]
      uncommon=nids[np.arange(2),count[inv.reshape(2,3)].argmin(axis=1)]
      common=np.setdiff1d(nids,uncommon)
      loc_common=np.abs(nids.T-common).argmin(axis=0)
      nid_selected[i,np.arange(2),loc_common]=uncommon[::-1]
      new_edge[i]=uncommon
    new_connects[fid_selected]=nid_selected
    #fid_updated=np.unique(map_e2f[eid_trg])
    #u_edge[eid_selected]=new_edge
    #aspect,eid_longest,min_max_ratio=get_aspect(new_connects,coords,u_edge,map_f2e,fid_updated)
    
  return new_connects
