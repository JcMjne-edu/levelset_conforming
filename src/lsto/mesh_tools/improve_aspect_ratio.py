import numpy as np

def get_mappings(connect):
  """
  connect : (m,3)
  """
  edge=connect[:,[[0,1],[1,2],[2,0]]] #(m,3,2)
  edge=np.sort(edge,axis=2) #(m,3,2)
  u_edge,inv,counts=np.unique(edge.reshape(-1,2),axis=0,return_inverse=True,return_counts=True) #(n,2), (m*3,), (n,)
  if (counts!=2).any():
    raise ValueError('Holes in mesh')
  map_e2f=np.arange(connect.shape[0]).repeat(3)[np.argsort(inv)].reshape(-1,2) #(n,2)
  return u_edge, map_e2f

def msk_edge_angle_updated(nids,coord,threshold_angle):
  """
  nids : (n,2,3)
  coord : (m,3)
  """
  vs=coord[nids] #(n,2,3,3)
  if (nids==np.roll(nids,1,axis=2)).sum():
    raise ValueError('Invalid nids')
  norm=np.cross(vs[:,:,1]-vs[:,:,0],vs[:,:,2]-vs[:,:,0]) #(n,2,3)
  n=np.linalg.norm(norm,axis=-1,keepdims=True) #(n,2,3)
  msk_n=(n!=0.0).all(axis=(-1,-2)) #(n,)
  norm=norm/np.where(n==0.,1.,n) #(n,2,3)
  cosines=(norm[:,0]*norm[:,1]).sum(axis=1) #(n,)
  msk=(cosines>np.cos(np.deg2rad(threshold_angle)))*msk_n
  return msk

def get_new_face(nids,coord,threshold_angle=1.0):
  """
  nids : (n,2,3)
  map_e2f : (n,2)
  """
  ar_min_init=get_min_aspect_ratio(nids,coord)
  diff0=nids[:,0]-nids[:,1] #(n2,3)
  diff1=nids[:,0]-np.roll(nids[:,1],1,axis=1) #(n2,3)
  diff2=nids[:,0]-np.roll(nids[:,1],2,axis=1) #(n2,3)
  diff=(np.concatenate([diff0,diff1,diff2],axis=1)==0)
  idx=np.argsort(np.abs(diff),axis=1)[:,-2:]
  i1=np.array([0,1,2,0,1,2,0,1,2])[idx]
  i2=np.array([0,1,2,2,0,1,1,2,0])[idx]

  m1=diff.reshape(-1,3,3)
  m2=np.zeros_like(m1)
  m2[:,np.arange(3).repeat(3),np.array([0,1,2,2,0,1,1,2,0])]=diff
  m1=m1.any(axis=1)
  m2=m2.any(axis=1)
  uncommon1=np.argmin(m1,axis=1)
  uncommon2=np.argmin(m2,axis=1)
  ar=np.arange(i2.shape[0])
  nids_new=nids.copy() #(n2,2,3)
  nids_new[ar,1,i2[:,0]]=nids[ar,0,uncommon1]
  nids_new[ar,0,i1[:,1]]=nids[ar,1,uncommon2]
  ar_min_updated=get_min_aspect_ratio(nids_new,coord)
  center=(coord[nids[ar,0,uncommon1]]+coord[nids[ar,1,uncommon2]])/2 #(n2,3)
  is_inside=is_inside_tris(center,coord[nids]) #(n2,)
  msk_angle=msk_edge_angle_updated(nids_new,coord,threshold_angle)
  msk=(ar_min_updated>ar_min_init)*is_inside*msk_angle
  return nids_new,msk

def get_min_aspect_ratio(nids,coord):
  """
  nids : (m,2,3)
  coord : (n,3)
  """
  vs=coord[nids] #(m,2,3,3)
  l=np.linalg.norm(vs-np.roll(vs,1,axis=2),axis=3) #(m,2,3)
  area=np.linalg.norm(np.cross(vs[:,:,0]-vs[:,:,1],vs[:,:,0]-vs[:,:,2]),axis=2) #(m,2)
  ar=area/(l.max(axis=2)**2) #(m,2)
  ar_min=ar.min(axis=1) #(m,)
  return ar_min

def get_aspect_ratio(connect,coord):
  vs=coord[connect] #(m,3,3)
  l=np.linalg.norm(vs-np.roll(vs,1,axis=1),axis=2) #(m,3)
  area=np.linalg.norm(np.cross(vs[:,0]-vs[:,1],vs[:,0]-vs[:,2]),axis=1) #(m,)
  ar=area/(l.max(axis=1)**2) #(m,)
  return ar

def get_independent_edge(eid,u_edge,nnode):
  """
  eid : (n,)
  u_edge : (m,2)
  """
  idx=np.arange(eid.shape[0])
  nid_used=np.zeros(nnode,dtype=bool)
  msk_eid_trg=np.zeros(eid.shape[0],dtype=bool)
  np.random.shuffle(idx)
  for i in idx:
    if not (nid_used[u_edge[eid[i]]]).any():
      nid_used[u_edge[eid[i]]]=True
      msk_eid_trg[i]=True
  #print(msk_eid_trg.sum()*2/np.unique(u_edge[eid[msk_eid_trg]]).shape[0])
  return msk_eid_trg

def is_inside_tris(v_trg,vs):
  """
  v_trg : (n,3)
  vs : (n,2,3,3)
  """
  norms=np.cross(vs[:,:,1]-vs[:,:,0],vs[:,:,2]-vs[:,:,0]) #(n,3)
  d=vs-v_trg[:,None,None] #(n,2,3,3)
  s1=np.sign((norms*np.cross(d[:,:,0],d[:,:,1])).sum(axis=2)) #(n,2)
  s2=np.sign((norms*np.cross(d[:,:,1],d[:,:,2])).sum(axis=2)) #(n,2)
  s3=np.sign((norms*np.cross(d[:,:,2],d[:,:,0])).sum(axis=2)) #(n,2)
  is_inside=(s1>0)*(s2>0)*(s3>0)+(s1<0)*(s2<0)*(s3<0) #(n,2)
  is_inside=is_inside.any(axis=1) #(n,)
  return is_inside


def improve_aspect(connect,coord,threshold_angle=1.0):
  """
  connect : (m,3)
  coord : (n,3)
  """
  connect=connect.copy()
  nnode=coord.shape[0]
  while True:
    u_edge,map_e2f=get_mappings(connect)
    ar=get_aspect_ratio(connect,coord)
    eid_trg=np.where(ar[map_e2f].min(axis=1)<0.2)[0]
    #print(eid_trg.shape,)
    #msk_angle=msk_edge_angle(connect,coord,map_e2f,threshold_angle)
    #eids=np.where(msk_angle)[0]
    #nids=(connect[map_e2f[eids]]).copy() #(n2,2,3)
    nids=(connect[map_e2f[eid_trg]]).copy() #(n2,2,3)
    nid_new,msk_update=get_new_face(nids,coord,threshold_angle)
    if not msk_update.any():
      break
    eid_valid=eid_trg[msk_update]
    #eid_valid=np.where(msk_update)[0]
    nid_new_valid=nid_new[msk_update]
    msk_eid_trg=get_independent_edge(eid_valid,u_edge,nnode)
    #print(msk_angle.sum(),msk_update.sum(),msk_eid_trg.sum())
    eid_trg=eid_valid[msk_eid_trg]
    nid_trg=nid_new_valid[msk_eid_trg]
    connect[map_e2f[eid_trg].flatten()]=nid_trg.reshape(-1,3)
  return connect