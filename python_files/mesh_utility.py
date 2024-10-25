from jax import jit,vmap
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

@jit
def point_in_hole(vert):
  """
  vert : (nface,, 3 3) float
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
  label=np.arange(nnode)
  edges=connect[:,[0,1,1,2,2,0]].reshape(-1,2)
  edges=np.sort(edges,axis=1)
  edges=np.unique(edges,axis=0)
  while True:
    _label=label.copy()
    le=label[edges]
    msk=(le[:,0]!=le[:,1])
    e=edges[msk]
    lemin=(label[e]).min(axis=1)
    label[e[:,0]]=(lemin)
    label[e[:,1]]=(lemin)
    if (_label==label).all():
      break
  _,label=np.unique(label,return_inverse=True)
  return label

def points_in_holes(coord,connect):
  if connect.shape[0]==0:
    return jnp.array([[0.,-1.,0.]])
  label=label_hole(np.asarray(connect))
  label_face=label[connect[:,0]]
  unique_label=jnp.arange(label.max()+1)
  verts=_return_vert(unique_label,label_face,connect,coord)
  points=vmap(point_in_hole)(verts)
  return np.array(points)

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

def visualize_ls(coords,connects):
  fig=go.Figure()
  fig.add_trace(go.Mesh3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],i=connects[:,0],j=connects[:,1],k=connects[:,2],opacity=0.5))
  fig.update_layout(scene=dict(aspectmode='data'))
  return fig

def visualize_mesh(connect,coord,opacity=0.5,render_edge=True):
  """
  connect: (n,4)
  coord: (n,3)
  """
  faces=connect[:,[0,1,2,0,2,3,0,3,1,1,3,2]].reshape(-1,3) #(4n,3)
  faces_sorted=np.sort(faces,axis=1)
  _,inv,counts=np.unique(faces_sorted,return_inverse=True,return_counts=True,axis=0)
  counts=counts[inv]
  face_valid=faces[counts==1] #(l,3)
  edge=face_valid[:,[0,1,0,2,1,2]].reshape(-1,2) #(3l,2)
  edge=np.sort(edge,axis=1)
  edge_unique=np.unique(np.sort(edge,axis=1),axis=0) #(k,2)
  edge_coord=coord[edge_unique] #(k,2,3)
  nones=np.array([None]*edge_coord.shape[0]*3).reshape(-1,1,3)
  edge_coord=np.concatenate([edge_coord,nones],axis=1).reshape(-1,3)  #(3k,3)
  fig=go.Figure()
  fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=face_valid[:,0],j=face_valid[:,1],k=face_valid[:,2],opacity=opacity))
  if render_edge:
    fig.add_trace(go.Scatter3d(x=edge_coord[:,0],y=edge_coord[:,1],z=edge_coord[:,2],mode='lines'))
  fig.update_layout(scene=dict(aspectmode='data'),margin=dict(l=0,r=0,b=0,t=10))
  return fig