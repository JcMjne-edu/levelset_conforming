from jax import jit,vmap
import jax.numpy as jnp
import numpy as np

@jit
def point_in_hole(vert):
  """
  coord : (nnode,3)
  connect : (nface,3)
  """
  normal=jnp.cross(vert[:,1]-vert[:,0],vert[:,2]-vert[:,0]) # (nface,3)
  fid_trg=jnp.linalg.norm(normal,axis=1).argmax()
  center=vert.mean(axis=1) # (nface,3)
  center_trg=center[fid_trg] # (3,)
  normal_normed=normal/jnp.linalg.norm(normal,axis=1,keepdims=True) # (nface,3)
  normal_trg_normed=normal_normed[fid_trg] # (3,)
  plane_const=(normal_normed*center).sum(axis=1) # (nface,)
  length=plane_const-normal_normed@center_trg # (nface,)
  inner=normal_normed@normal_trg_normed # (nface,)
  msk_invalid_l=(inner==0.0).at[fid_trg].set(True)
  length=length/inner # (nface,)
  msk_invalid_l=msk_invalid_l+(length<=0.)
  intercept=center_trg+length[:,None]*normal_trg_normed # (nface,3)
  vec=vert-intercept[:,None] # (nface,3,3)
  cross1=jnp.cross(vec[:,0],vec[:,1]) # (nface,3)
  cross2=jnp.cross(vec[:,1],vec[:,2]) # (nface,3)
  cross3=jnp.cross(vec[:,2],vec[:,0]) # (nface,3)
  sign1=(cross1*cross2).sum(axis=1) # (nface,)
  sign2=(cross2*cross3).sum(axis=1) # (nface,)
  sign3=(cross3*cross1).sum(axis=1) # (nface,)
  msk_sign=(sign1>0.)*(sign2>0.)*(sign3>0.)+(sign1<=0.)*(sign2<=0.)*(sign3<=0.)
  msk_invalid_l=msk_invalid_l+(~msk_sign)
  length=jnp.where(msk_invalid_l,jnp.nan,length)
  fid_counterpart=jnp.nanargmin(length)
  point=center_trg+0.5*length[fid_counterpart]*normal_trg_normed
  return point

def label_hole(connect):
  nnode=connect.max()+1
  label=jnp.arange(nnode)
  edges=connect[:,[0,1,1,2,2,0]].reshape(-1,2)
  edges=jnp.sort(edges,axis=1)
  edges=jnp.unique(edges,axis=0)
  while True:
    label,if_stop=_update_label(edges,label)
    if if_stop:
      break
  _,label=jnp.unique(label,return_inverse=True)
  return label

@jit
def _update_label(edges,label):
  _label=label.at[edges[:,0]].set(label[edges].min(axis=1))
  _label=_label.at[edges[:,1]].set(_label[edges].min(axis=1))
  if_stop=(label==_label).all()
  return _label,if_stop

def points_in_holes(coord,connect):
  label=label_hole(connect)
  label_face=label[connect[:,0]]
  unique_label=jnp.unique(label)
  verts=_return_vert(unique_label,label_face,connect,coord)
  points=vmap(point_in_hole)(verts)
  return points

def _return_vert(unique_label,label_face,connect,coord):
  counts=jnp.bincount(label_face)
  n=counts.max()
  verts=coord[connect]
  iz=jnp.argsort(label_face)
  iy=jnp.concatenate([jnp.arange(i) for i in counts])
  ix=label_face[iz]
  out=jnp.zeros((unique_label.shape[0],n,3,3))
  out=out.at[ix,iy].set(verts[iz])
  return out

def is_on_edge(vt,nid_global,node,edge,threashold=1e-4):
  """
  vt : (n_trg,3)
  node : (n_ref,3)
  edge : (n_e,2)
  """
  vert_edge=node[edge] #(n_e,2,3)
  v12=vert_edge[:,1]-vert_edge[:,0] #(n_e,3)
  length_edge=jnp.linalg.norm(v12,axis=1) #(n_e,)
  v1t=vt[:,None]-vert_edge[:,0] #(n_trg,n_e,3)
  metric=(v12*v1t).sum(axis=2)/length_edge**2 #(n_trg,n_e)
  is_between=(metric>0.)*(metric<1.)
  dist_to_edge=jnp.linalg.norm(jnp.cross(v1t,v12),axis=2)/length_edge #(n_trg,n_e)
  is_close=dist_to_edge<threashold
  is_on_edge=jnp.any(is_between*is_close,axis=1)
  
  nid_on_edge=jnp.where(is_on_edge)[0] #(n_on_edge,)
  eid_trg=dist_to_edge[nid_on_edge].argmin(axis=1) #(n_on_edge,)
  metric_trg=metric[nid_on_edge,eid_trg] #(n_on_edge,)
  weight=jnp.array([1.-metric_trg,metric_trg]).T.flaten() #(n_on_edge*2)
  indices1=nid_global[nid_on_edge].repeat(2) #(n_on_edge*2)
  indices2=edge[eid_trg].flatten() #(n_on_edge*2)
  indices=jnp.stack([indices1,indices2],axis=1) #(n_on_edge*2,2)
  return indices,weight,nid_global[nid_on_edge]

def insideout(vert,connect):
  """
  vert:(ne,4,3)
  connect:(ne,4)
  """
  v1=vert[:,1]-vert[:,0] #(ne,3)
  v2=vert[:,2]-vert[:,0] #(ne,3)
  v3=vert[:,3]-vert[:,0] #(ne,3)
  trg=(np.cross(v1,v2)*v3).sum(axis=1) #(ne,)
  trg=np.sign(trg) #(ne,)
  if (trg==0).sum():
    print(f'zero volume tetra elements detected ({(trg==0).sum()})')
    print(connect[np.where(trg==0)[0]])
  normal=np.take(connect,np.where(trg>0)[0],axis=0) #(ne_n,4)
  inv=connect[np.where(trg<0)[0]] #(ne_i,4)
  inv=inv[:,[1,0,2,3]] #(ne_i,4)
  out=np.concatenate((normal,inv)) #(ne,4)
  return out

def check(conect,coord):
  vert=coord[conect]
  v1=vert[:,1]-vert[:,0]
  v2=vert[:,2]-vert[:,0]
  v3=vert[:,3]-vert[:,0]
  trg=(np.cross(v1,v2)*v3).sum(axis=1)
  return trg
