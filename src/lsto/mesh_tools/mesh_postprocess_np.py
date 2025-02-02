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
  connects=connects.copy()
  coords=coords.copy()
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
  if (counts!=2).any():
    raise ValueError('Holes in mesh')
    #print('Holes in mesh')
  map_f2e=inv.reshape(-1,3) #(m,3)
  #mapping from edge to face
  map_e2f=np.arange(connects.shape[0]).repeat(3)[np.argsort(inv)].reshape(-1,2) #(m,2)
  eid_longest=map_f2e[np.arange(len(edge)),l_argmax] #(m,)
  return aspect,eid_longest,min_max_ratio,u_edge,map_e2f,map_f2e

def extract_unique_eid_trg(eid_trg,u_edge,nnode):
  edge_trg=u_edge[eid_trg] #(m,2)
  nid_used=np.zeros(nnode,dtype=bool)
  unique_edge_id=np.zeros(len(eid_trg),dtype=bool)
  for i,e in enumerate(edge_trg):
    if not nid_used[e].any():
      nid_used[e]=True
      unique_edge_id[i]=True
  return unique_edge_id

def eliminate_lowaspect_triangle(connects,coords,threshold=1e-3):
  new_connects=connects.copy()
  nnode=coords.shape[0]
  for i in range(10):
    aspect,eid_longest,min_max_ratio,u_edge,map_e2f,map_f2e=setup(new_connects,coords)
    if aspect.min()>threshold:
      break
    msk_trg=(aspect<threshold)*(min_max_ratio>0.05)
    if msk_trg.sum()==0:
      #print('2')
      break
    _aspect_trg=aspect[msk_trg] #(mt,)
    _eid_trg=eid_longest[msk_trg] #(mt,)
    _fid_trg=map_e2f[_eid_trg] #(mt,2)
    _nid_trg=new_connects[_fid_trg] #(mt,2,3)
    _new_nids,min_aspect_updated,isinside=get_new_nids(_nid_trg,coords)
    #print(isinside.dtype)
    msk_trg_update=(_aspect_trg<min_aspect_updated)#*isinside
    if msk_trg_update.sum()==0:
      #print('3')
      break

    _fid_trg=_fid_trg[msk_trg_update]
    _nid_trg=_nid_trg[msk_trg_update]
    _new_nids=_new_nids[msk_trg_update]
    _eid_trg=_eid_trg[msk_trg_update]

    idx_shuffle=np.arange(_fid_trg.shape[0])
    np.random.shuffle(idx_shuffle)
    eid_trg=_eid_trg[idx_shuffle]
    _fid_trg=_fid_trg[idx_shuffle]
    _nid_trg=_nid_trg[idx_shuffle]
    _new_nids=_new_nids[idx_shuffle]

    unique_edge_id=extract_unique_eid_trg(eid_trg,u_edge,nnode) #(ms,)
    #print(eid_trg.shape,unique_edge_id.shape)
    fid_selected=_fid_trg[unique_edge_id] #(ms,2)
    new_connects[fid_selected]=_new_nids[unique_edge_id]
  #aspect,eid_longest,min_max_ratio,u_edge,map_e2f,map_f2e=setup(new_connects,coords)  
  return new_connects#,isinside,_nid_trg,_aspect_trg

def get_new_nids(nid_trg,coords):
  """
  nid_trg : (ms,2,3)
  coords : (n,3)
  """
  nid_selected=nid_trg.copy()
  min_aspect=np.zeros(nid_selected.shape[0]) #(ms,)
  isinside=np.zeros(nid_selected.shape[0],dtype=bool)
  for i,nids in enumerate(nid_selected):
    _,inv,count=np.unique(nids,return_inverse=True,return_counts=True)
    count[inv.reshape(2,3)]
    uncommon=nids[np.arange(2),count[inv.reshape(2,3)].argmin(axis=1)]
    common=np.setdiff1d(nids,uncommon)
    loc_common=np.abs(nids.T-common).argmin(axis=0)
    nid_selected[i,np.arange(2),loc_common]=uncommon[::-1]
    aspects=calc_aspect(coords[nid_selected[i]]) #(2,3,3)
    coord_center=coords[uncommon].mean(axis=0) #(3,)
    isinside[i]=is_inside_tris(coord_center,coords[nid_trg[i]]).any()
    min_aspect[i]=aspects.min()
  return nid_selected,min_aspect,isinside

def calc_aspect(vs):
  """
  vs : (m,3,3)
  """
  area=np.linalg.norm(np.cross(vs[:,1]-vs[:,0],vs[:,2]-vs[:,0]),axis=1)
  l=np.linalg.norm(vs-np.roll(vs,1,axis=1),axis=2) #(m,3)
  aspect=area/l.max(axis=1)**2 #(m,)
  return aspect

def is_inside_tris(v_trg,vs):
  """
  v_trg : (3,)
  vs : (m,3,3)
  """
  norms=np.cross(vs[:,1]-vs[:,0],vs[:,2]-vs[:,0]) #(m,3)
  d=vs-v_trg # (m,3,3)
  s1=np.sign((norms*np.cross(d[:,0],d[:,1])).sum(axis=1)) #(m,)
  s2=np.sign((norms*np.cross(d[:,1],d[:,2])).sum(axis=1)) #(m,)
  s3=np.sign((norms*np.cross(d[:,2],d[:,0])).sum(axis=1)) #(m,)
  is_inside=(s1>0)*(s2>0)*(s3>0)+(s1<0)*(s2<0)*(s3<0) #(m,)
  return is_inside