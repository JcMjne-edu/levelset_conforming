from jax import jit
from jax.lax import stop_gradient
import jax.numpy as jnp
import numpy as np
from lsto.stl_tools import stl_from_connect_and_coord

#@jit
def area_from_coord_connect_jx(coord,connect):
  """
  cooord : (n,3)
  connect : (m,3)
  """
  v=coord[connect] # (m,3,3)
  e=v-jnp.roll(v,1,axis=1) # (m,3,3)
  area=jnp.linalg.norm(jnp.cross(e[:,0],e[:,1]),axis=1) # (m,)
  return area

#@jit
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
    #
    msk_l=(ledge<thresh_l) # (m,)
    if msk_l.sum()==0:
      break
    eid_invalid=eid[msk_l] # (m,2)
    ledge_invalid=ledge[msk_l] # (m,)
    #
    sort_idx=np.argsort(ledge_invalid) # (m,)
    eid_invalid=eid_invalid[sort_idx] # (m,2)
    
    nids_used=np.zeros(nnode,dtype=bool)
    edge_trg=np.zeros(eid_invalid.shape[0],dtype=bool)
    edge_trg=_edge_trg(nids_used,edge_trg,eid_invalid)
    _coord_additional=_coord[edge_trg].mean(axis=1) # (m,3)
    msk_root=(_coord[edge_trg][:,:,1]==0.0).any(axis=1) # (m,)
    _coord_additional=_coord_additional.at[msk_root,1].set(0.0)
    _coord=_coord.at[edge_trg[:,0]].set(_coord_additional)
    
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

def mesh_postprocess_jx(connect,coord,thresh_l=1e-2):
  """
  threash_l : threashold length within which nodes are merged
  threash_v : threashold height ratio within which tetrahedra elements are collapsed
  """
  stl_from_connect_and_coord(connect,stop_gradient(coord)).save('./stl/check_original.stl')
  _coord,_connect=merge_close_nodes_jx(coord,connect,thresh_l)
  _connect=remove_zero_area_triangles(_connect)
  stl_from_connect_and_coord(_connect,stop_gradient(_coord)).save('./stl/check_node_merged.stl')
  #_coord,_connect=merge_flat_tris_jx(_coord,_connect,thresh_v)
  #stl_from_connect_and_coord(_connect,stop_gradient(_coord)).save('./stl/check_flat.stl')
  _connect=remove_zero_area_triangles(_connect)
  _connect=check_duplicate(_connect)
  unid,_connect=jnp.unique(_connect,return_inverse=True)
  _coord=_coord[unid]
  return _coord,_connect

def remove_zero_area_triangles(connects):
  """
  connects: np.array, shape=(n_triangles, 3)
  """
  idx_valid=((jnp.roll(connects,1,axis=1)-connects)!=0).all(axis=1)
  
  return connects[idx_valid]

def resolve_flattened_region(connect,coord):
  """
  connect : (m,3)
  coord : (n,3)
  """
  max_iter=20
  for i in range(max_iter):
    # Get unique edges and their mappings
    edge=connect[:,[[0,1],[1,2],[2,0]]] #(m,3,2)
    edge=jnp.sort(edge,axis=2) #(m,3,2)
    u_edge,inv,counts=jnp.unique(edge.reshape(-1,2),axis=0,return_inverse=True,return_counts=True) #(n,2), (m*3,), (n,)
    if (counts==1).any():
      raise ValueError('Holes in mesh')
    map_e2f=jnp.arange(connect.shape[0]).repeat(3)[jnp.argsort(inv)].reshape(-1,2) #(n,2)


    vs=coord[connect] #(m,3,3)
    norm=jnp.cross(vs[:,1,:]-vs[:,0,:],vs[:,2,:]-vs[:,0,:],axis=1) # (m,3)
    norm=norm/jnp.linalg.norm(norm,axis=1,keepdims=True) #(m,3)
    norm_pair=norm[map_e2f] #(ne,2,3)
    angle=(norm_pair[:,0]*norm_pair[:,1]).sum(axis=1) #(ne,)
    edge_trg=jnp.where(angle+1.0<1e-5)[0] #(ne_trg,)
    if edge_trg.size==0:
      return connect, coord
    v_edge_trg=coord[u_edge[edge_trg]] #(ne_trg,2,3)
    edge_length_trg=jnp.linalg.norm(v_edge_trg[:,1,:]-v_edge_trg[:,0,:],axis=1) #(ne_trg,)
    # Sort edges by length
    edge_arg=jnp.argsort(edge_length_trg) #(ne_trg,)
    used_node=[]
    eid_collapse=[]
    print(edge_trg[edge_arg])
    for eid in edge_trg[edge_arg]:
      nids=u_edge[eid] #(2,)
      print(eid)
      if (jnp.isin(nids,jnp.array(used_node))).any():
        continue
      used_node.extend(nids.tolist())
      eid_collapse.append(eid)
    nid_merge=u_edge[jnp.array(eid_collapse)] #(n_collapse,2)
    midpoint=coord[nid_merge].mean(axis=1) #(n_collapse,3)
    coord=coord.at[nid_merge[:,0]].set(midpoint)
    msk=jnp.ones(coord.shape[0],bool)
    msk=msk.at[nid_merge[:,1]].set(False)
    #coord=coord[msk]
    msk_connect_remain=(msk[connect].sum(axis=1)>=2)
    msk_connect_replace=(msk[connect].sum(axis=1)==2)
    connect_replace=connect[msk_connect_replace] #(m2,3)
    for nid_remain,nid_del in nid_merge:
      connect_replace=jnp.where(connect_replace==nid_del,nid_remain,connect_replace)
    connect=connect.at[msk_connect_replace].set(connect_replace)
    connect=connect[msk_connect_remain]
    unid,inv=jnp.unique(connect,return_inverse=True)
    connect=inv.reshape(connect.shape)
    coord=coord[unid]
    print(connect.max(),coord.shape)
    stl_from_connect_and_coord(connect,stop_gradient(coord)).save('./stl/check_flattened_'+str(i)+'.stl')
  return connect, coord