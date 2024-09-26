from jax import jit
from jax.lax import stop_gradient
import jax.numpy as jnp
from numba import jit as njit
import numpy as np

@jit
def area_from_coord_connect_jx(coord,connect):
  """
  cooord : (n,3)
  connect : (m,3)
  """
  v=coord[connect] # (m,3,3)
  e=v-jnp.roll(v,1,axis=1) # (m,3,3)
  area=jnp.linalg.norm(jnp.cross(e[:,0],e[:,1]),axis=1) # (m,)
  return area

@jit
def _metric(coord,connect):
  """
  cooord : (n,3)
  connect : (m,3)
  """
  v=coord[connect] # (m,3,3)
  e=v-jnp.roll(v,1,axis=1) # (m,3,3)
  area=jnp.linalg.norm(jnp.cross(e[:,0],e[:,1]),axis=1) # (m,)
  l_edge=jnp.linalg.norm(e,axis=2)
  max_edge=jnp.max(l_edge,axis=1) # (m,)
  h=area/max_edge # (m,)
  val=h/max_edge # (m,)
  return val,l_edge

def merge_close_nodes_jx(coord,connect,thresh_l=1e-2):
  _connect=connect#np.array(connect)
  _coord=coord
  nnode=_coord.shape[0]
  connect_id_changed=np.ones(_connect.shape[0],bool)
  while True:
    eid=_connect[connect_id_changed][:,[0,1,1,2,2,0]].reshape(-1,2) # (m,2)
    eid=np.sort(eid,axis=1) 
    eid=np.unique(eid,axis=0)
    ledge=np.linalg.norm(stop_gradient(_coord[eid[:,0]]-_coord[eid[:,1]]),axis=1) # (m,)
    #print('2.5',ledge.shape)
    msk_l=(ledge<thresh_l) # (m,)
    if msk_l.sum()==0:
      break
    eid_invalid=eid[msk_l] # (m,2)
    ledge_invalid=ledge[msk_l] # (m,)
    #print('3')
    sort_idx=np.argsort(ledge_invalid) # (m,)
    eid_invalid=eid_invalid[sort_idx] # (m,2)
    
    #nids_used=[]
    #edge_trg=[]
    #print('2',type(eid_invalid),eid_invalid.shape)
    #print(eid_invalid)
    #print(((eid_invalid[:,0]-eid_invalid[:,1])==0).sum())
    #for eid in eid_invalid:
    #  if not np.isin(eid,nids_used).any():
    #    edge_trg.append(eid)
    #    nids_used+=[*eid]
    #print('3')
    
    #print(eid_invalid.shape)
    #nids_used=np.zeros(_coord.shape[0],dtype=bool)
    #edge_trg=np.zeros(eid_invalid.shape[0],dtype=bool)
    #for i,eid in enumerate(eid_invalid):
    #  if not nids_used[eid].any():
    #    edge_trg[i]=True
    #    nids_used[eid]=True
    ##edge_trg=np.array(edge_trg) # (m,2)
    #edge_trg=eid_invalid[edge_trg] # (m,2)
    
    nids_used=np.zeros(nnode,dtype=bool)
    edge_trg=np.zeros(eid_invalid.shape[0],dtype=bool)
    edge_trg=_edge_trg(nids_used,edge_trg,eid_invalid)
    _coord_additional=_coord[edge_trg].mean(axis=1) # (m,3)
    msk_root=(_coord[edge_trg][:,:,1]==0.0).any(axis=1) # (m,)
    _coord_additional=_coord_additional.at[msk_root,1].set(0.0)
    _coord=_coord.at[edge_trg[:,0]].set(_coord_additional)
    #print('start')
    #replace_dict=dict(zip(edge_trg[:,1],edge_trg[:,0]))
    #print(edge_trg.shape,edge_trg.max(),_connect.max(),_coord.shape)
    #_connect=np.array([replace_dict.get(nid,nid) for nid in _connect.flatten()]).reshape(_connect.shape)
    msk_replace=np.zeros(_coord.shape[0],dtype=bool)
    msk_replace[edge_trg.flatten()]=True
    connect_id_changed=msk_replace[_connect].any(axis=1)
    _connect=_replace_nid(nnode,edge_trg,_connect)
    area=area_from_coord_connect_jx(stop_gradient(_coord),jnp.array(_connect))
    _connect=_connect[area>0.0]
    connect_id_changed=connect_id_changed[area>0.0]
  return _coord,_connect

#@njit
def _edge_trg(nids_used,edge_trg,eid_invalid):
  for i,eid in enumerate(eid_invalid):
    if not nids_used[eid].any():
      edge_trg[i]=True
      nids_used[eid]=True
  #edge_trg=np.array(edge_trg) # (m,2)
  edge_trg=eid_invalid[edge_trg] # (m,2)
  return edge_trg

def merge_flat_tris_jx(coord,connect,thresh_v=1e-2):
  _connect=np.array(connect)
  _coord=coord
  nnode=_coord.shape[0]
  while True:
    val,l_edge=_metric(stop_gradient(_coord),_connect) # (m,)
    msk_trg=(val<=thresh_v)
    if not msk_trg.any():
      break
    l_edge_trg=l_edge.min(axis=1)[msk_trg] 
    nid1=l_edge[msk_trg].argmin(axis=1)
    nid2=(nid1+2)%3
    eid_invalid=np.stack([_connect[msk_trg,nid1],_connect[msk_trg,nid2]],axis=1) # (m,2)
    eid_invalid=eid_invalid[np.argsort(l_edge_trg)]
    
    #nids_used=[]
    #edge_trg=[]
    #for eid in eid_invalid:
    #  if not np.isin(eid,nids_used).any():
    #    edge_trg.append(eid)
    #    nids_used+=[*eid]
    #edge_trg=np.array(edge_trg)
   
    #nids_used=np.zeros(_coord.shape[0],dtype=bool)
    #edge_trg=np.zeros(eid_invalid.shape[0],dtype=bool)
    #for i,eid in enumerate(eid_invalid):
    #  if not nids_used[eid].any():
    #    edge_trg[i]=True
    #    nids_used[eid]=True
    ##edge_trg=np.array(edge_trg) # (m,2)
    #edge_trg=eid_invalid[edge_trg] # (m,2)
    
    nids_used=np.zeros(nnode,dtype=bool)
    edge_trg=np.zeros(eid_invalid.shape[0],dtype=bool)
    edge_trg=_edge_trg(nids_used,edge_trg,eid_invalid)
    
    _coord_additional=_coord[edge_trg].mean(axis=1) # (m,3)
    msk_root=(_coord[edge_trg][:,:,1]==0.0).any(axis=1) # (m,)
    _coord_additional=_coord_additional.at[msk_root,1].set(0.0)
    _coord=_coord.at[edge_trg[:,0]].set(_coord_additional)
    _connect=_replace_nid(nnode,edge_trg,_connect)
    area=area_from_coord_connect_jx(stop_gradient(_coord),jnp.array(_connect))
    _connect=_connect[area>0.0]
  return _coord,_connect

def _replace_nid(nnode,edge_trg,connect):
  map_replace=np.arange(nnode)
  map_replace[edge_trg[:,1]]=edge_trg[:,0]
  return map_replace[connect]


def check_duplicate(connect):
  connect_sorted=np.sort(connect,axis=1)
  _,inverse,count=np.unique(connect_sorted,axis=0,return_inverse=True,return_counts=True)
  msk=(count==1)[inverse]
  connect=connect[msk]

  unid,counts=np.unique(connect,return_counts=True)
  nid_invalid=unid[np.where(counts==1)[0]]
  msk=np.isin(connect,nid_invalid).any(axis=1)
  connect=connect[~msk]
  return connect

def mesh_postprocess_jx(coord,connect,thresh_l=1e-2,thresh_v=1e-2):
  """
  threash_l : threashold length within which nodes are merged
  threash_v : threashold height ratio within which tetrahedra elements are collapsed
  """
  _coord,_connect=merge_close_nodes_jx(coord,connect,thresh_l)
  _coord,_connect=merge_flat_tris_jx(_coord,_connect,thresh_v)
  _connect=check_duplicate(_connect)
  unid,_connect=jnp.unique(_connect,return_inverse=True)
  _coord=_coord[unid]
  return _coord,_connect
