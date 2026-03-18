import jax.numpy as jnp
from jax.lax import stop_gradient
import numpy as np
from lsto.levelset_conforming_penetrate import *

def isinside_tri_np(foot,tri):
  """
  Check if a point is inside a triangle.

  Parameters
  ----------
  foot : (M, N, 3) array_like
      The coordinates of the point.
  tri : (N, 3, 3) array_like
      The coordinates of the vertices of the triangle.

  Returns
  -------
  inside : bool
      True if the point is inside the triangle, False otherwise.
  """
  diff=tri-foot[:,:,None] # (M, N, 3, 3)
  e0 = np.cross(diff[:,:,0], diff[:,:,1]) # (M, N, 3)
  e1 = np.cross(diff[:,:,1], diff[:,:,2]) # (M, N, 3)
  e2 = np.cross(diff[:,:,2], diff[:,:,0]) # (M, N, 3)

  sign1=(e0*e1).sum(axis=-1) # (M, N)
  sign2=(e1*e2).sum(axis=-1) # (M, N)
  msk=(sign1>=0.0)*(sign2>=0.0) # (M, N)
  return msk

def ison_edge_np(foot,v_ref):
  """
  Check if a point is on the edge of a triangle.

  Parameters
  ----------
  foot : (M, N, 3) array_like
      The coordinates of the point.
  v_ref : (N, 2, 3) array_like
      The coordinates of the vertices of the triangle.

  Returns
  -------
  on_edge : bool
      True if the point is on the edge of the triangle, False otherwise.
  """
  d1=foot-v_ref[:,0] # (M, N, 3)
  d2=foot-v_ref[:,1] # (M, N, 3)
  d1/=np.linalg.norm(d1, axis=-1,keepdims=True)
  d2/=np.linalg.norm(d2, axis=-1,keepdims=True)
  msk=(d1*d2).sum(axis=-1)<=0.0 # (M, N)
  return msk

def get_closest_trimeshId_np(v,tri,norm,norm_edge1,norm_edge2,norm_edge3):
#def get_closest_trimeshId_np(v,tri,norm):
  """
  Compute the closest triangle to a point.

  Parameters
  ----------
  v : (M,3) array_like
      The point in 3D space.
  tri : (N, 3, 3) array_like
      The coordinates of the vertices of the triangles.
  norm : (N, 3) array_like
      The normals of the triangles in the mesh.

  Returns
  -------
  tid : (M,) array_like
      The index of the closest triangle to the point.
  """
  diff=v[:,None,None]-tri # (M, N, 3, 3)
  dist_node=np.linalg.norm(diff, axis=-1) # (M, N, 3)
  dist_surf=(diff[:,:,0]*norm).sum(axis=-1) # (M, N)
  foot_sruf=v[:,None]-dist_surf[:,:,None]*norm # (M, N, 3)
  #norm_edge1=np.cross(tri[:,1]-tri[:,0],norm) # (N, 3)
  #norm_edge1=norm_edge1/np.linalg.norm(norm_edge1, axis=-1,keepdims=True) # (N, 3)
  #norm_edge2=np.cross(tri[:,2]-tri[:,1],norm)
  #norm_edge2=norm_edge2/np.linalg.norm(norm_edge2, axis=-1,keepdims=True) # (N, 3)
  #norm_edge3=np.cross(tri[:,0]-tri[:,2],norm)
  #norm_edge3=norm_edge3/np.linalg.norm(norm_edge3, axis=-1,keepdims=True) # (N, 3)
  #print(norm_edge1.shape, norm_edge2.shape, norm_edge3.shape)

  dist_edge1=(diff[:,:,0]*norm_edge1).sum(axis=-1) # (M, N)
  dist_edge2=(diff[:,:,1]*norm_edge2).sum(axis=-1) # (M, N)
  dist_edge3=(diff[:,:,2]*norm_edge3).sum(axis=-1) # (M, N)

  foot_edge1=foot_sruf-dist_edge1[:,:,None]*norm_edge1 # (M, N, 3)
  foot_edge2=foot_sruf-dist_edge2[:,:,None]*norm_edge2 # (M, N, 3)
  foot_edge3=foot_sruf-dist_edge3[:,:,None]*norm_edge3 # (M, N, 3)

  msk_edge1=ison_edge_np(foot_edge1,tri[:,[0,1]]) # (M, N)
  msk_edge2=ison_edge_np(foot_edge2,tri[:,[1,2]]) # (M, N)
  msk_edge3=ison_edge_np(foot_edge3,tri[:,[2,0]]) # (M, N)

  dist_edge1=np.where(msk_edge1,dist_edge1,np.minimum(dist_node[:,:,0],dist_node[:,:,1])) # (M, N)
  dist_edge2=np.where(msk_edge2,dist_edge2,np.minimum(dist_node[:,:,1],dist_node[:,:,2])) # (M, N)
  dist_edge3=np.where(msk_edge3,dist_edge3,np.minimum(dist_node[:,:,2],dist_node[:,:,0])) # (M, N)

  dist_edge=np.minimum(np.abs(dist_edge1),np.abs(dist_edge2)) # (M, N)
  dist_edge=np.minimum(dist_edge,np.abs(dist_edge3)) # (M, N)
  dist_edge=np.sqrt(dist_edge**2+dist_surf**2) # (M, N)
  
  msk_isinside=isinside_tri_np(foot_sruf,tri) # (M, N)
  dist=np.where(msk_isinside,np.abs(dist_surf),dist_edge) # (M, N)
  #return dist
  arg_min_dist=np.abs(dist).argmin(axis=-1) # (M,)
  return arg_min_dist

def _nparray(arr):
  return np.asarray(stop_gradient(arr))

def get_norm(connect,coord):
  """
  Compute the normals of a triangle mesh.

  Parameters
  ----------
  tri : (N, 3, 3) array_like
      The coordinates of the vertices of the triangles.

  Returns
  -------
  norm : (N, 3) array_like
      The normals of the triangles.
  """
  tri=coord[connect] # (N, 3, 3)
  v0 = tri[:, 0, :]
  v1 = tri[:, 1, :]
  v2 = tri[:, 2, :]

  e0 = v1 - v0
  e1 = v2 - v0
  n = jnp.cross(e0, e1)
  n_norm = jnp.linalg.norm(n, axis=1, keepdims=True)

  return n / n_norm

def inside_tri(v_on_surf,tri):
  """
  Check if a point is inside a triangle.

  Parameters
  ----------
  v_on_surf : (M, 3) array_like
      The coordinates of the point.
  tri : (M, 3, 3) array_like
      The coordinates of the vertices of the triangle.

  Returns
  -------
  inside : bool
      True if the point is inside the triangle, False otherwise.
  """
  d=tri-v_on_surf[:,None] #(M,3,3)
  e0 = jnp.cross(d[:,0], d[:,1]) # (M,3)
  e1 = jnp.cross(d[:,1], d[:,2]) # (M,3)
  e2 = jnp.cross(d[:,2], d[:,0]) # (M,3)
  
  sign1=(e0*e1).sum(axis=-1) # (M,)
  sign2=(e1*e2).sum(axis=-1) # (M,)
  msk=(sign1>=0.0)*(sign2>=0.0) # (M,)
  return msk

def isinside_tri_jx(foot,tri):
  """
  Check if a point is inside a triangle.

  Parameters
  ----------
  foot : (M, 3) array_like
      The coordinates of the point.
  tri : (M, 3, 3) array_like
      The coordinates of the vertices of the triangle.

  Returns
  -------
  inside : bool
      True if the point is inside the triangle, False otherwise.
  """
  diff=tri-foot[:,None] # (M, 3, 3)
  e0 = jnp.cross(diff[:,0], diff[:,1]) # (M, 3)
  e1 = jnp.cross(diff[:,1], diff[:,2]) # (M, 3)
  e2 = jnp.cross(diff[:,2], diff[:,0]) # (M, 3)

  sign1=(e0*e1).sum(axis=-1) # (M,)
  sign2=(e1*e2).sum(axis=-1) # (M,)
  msk=(sign1>=0.0)*(sign2>=0.0) # (M,)
  return msk

def ison_edge_jx(foot,v_ref):
  """
  Check if a point is on the edge of a triangle.

  Parameters
  ----------
  foot : (M, 3) array_like
      The coordinates of the point.
  v_ref : (M, 2, 3) array_like
      The coordinates of the vertices of the triangle.

  Returns
  -------
  on_edge : bool
      True if the point is on the edge of the triangle, False otherwise.
  """
  d1=foot-v_ref[:,0] # (M, 3)
  d2=foot-v_ref[:,1] # (M, 3)
  d1/=jnp.linalg.norm(d1, axis=-1,keepdims=True)
  d2/=jnp.linalg.norm(d2, axis=-1,keepdims=True)
  msk=(d1*d2).sum(axis=-1)<=0.0 # (M,)
  return msk

def min_distance_to_trimesh(v,norm,tri):
  """
  Compute the minimum distance from a point to a triangle mesh.

  Parameters
  ----------
  v : (M,3) array_like
      The point in 3D space.
  v_ref : (N, 3) array_like
      The coordinates of the reference points of the triangle mesh.
  norm : (N, 3) array_like
      The normals of the triangles in the mesh.
  tri : (N, 3, 3) array_like

  Returns
  -------
  d : (M,) array_like
      The minimum distance from the point to the triangle mesh.
  """
  norm_edge1=jnp.cross(tri[:,1]-tri[:,0],norm) # (M, 3)
  norm_edge1=norm_edge1/jnp.linalg.norm(norm_edge1, axis=-1,keepdims=True) # (M, 3)
  norm_edge2=jnp.cross(tri[:,2]-tri[:,1],norm)
  norm_edge2=norm_edge2/jnp.linalg.norm(norm_edge2, axis=-1,keepdims=True) # (M, 3)
  norm_edge3=jnp.cross(tri[:,0]-tri[:,2],norm)
  norm_edge3=norm_edge3/jnp.linalg.norm(norm_edge3, axis=-1,keepdims=True) # (M, 3)

  arg_min_dist=get_closest_trimeshId_np(_nparray(v),_nparray(tri),_nparray(norm),
                                        _nparray(norm_edge1),_nparray(norm_edge2),_nparray(norm_edge3)) # (M,)
  tri_msked=tri[arg_min_dist] # (M, 3, 3)
  norm_masked=norm[arg_min_dist] # (M, 3)

  # Compute the squared distances from the point to each triangle
  diff=v[:,None]-tri_msked # (M, 3, 3)
  dist_node=jnp.linalg.norm(diff, axis=-1) # (M, 3)
  dist_surf=(diff[:,0]*norm_masked).sum(axis=-1) # (M,)
  foot_sruf=v-dist_surf[:,None]*norm_masked # (M, 3)

  dist_edge1=(diff[:,0]*norm_edge1[arg_min_dist]).sum(axis=-1) # (M,)
  dist_edge2=(diff[:,1]*norm_edge2[arg_min_dist]).sum(axis=-1) # (M,)
  dist_edge3=(diff[:,2]*norm_edge3[arg_min_dist]).sum(axis=-1) # (M,)

  foot_edge1=foot_sruf-dist_edge1[:,None]*norm_edge1[arg_min_dist] # (M, 3)
  foot_edge2=foot_sruf-dist_edge2[:,None]*norm_edge2[arg_min_dist] # (M, 3)
  foot_edge3=foot_sruf-dist_edge3[:,None]*norm_edge3[arg_min_dist] # (M, 3)

  msk_edge1=ison_edge_jx(foot_edge1,tri_msked[:,[0,1]]) # (M,)
  msk_edge2=ison_edge_jx(foot_edge2,tri_msked[:,[1,2]]) # (M,)
  msk_edge3=ison_edge_jx(foot_edge3,tri_msked[:,[2,0]]) # (M,)

  dist_edge1=jnp.where(msk_edge1,dist_edge1,jnp.minimum(dist_node[:,0],dist_node[:,1])) # (M,)
  dist_edge2=jnp.where(msk_edge2,dist_edge2,jnp.minimum(dist_node[:,1],dist_node[:,2])) # (M,)
  dist_edge3=jnp.where(msk_edge3,dist_edge3,jnp.minimum(dist_node[:,2],dist_node[:,0])) # (M,)

  dist_edge=jnp.minimum(jnp.abs(dist_edge1),jnp.abs(dist_edge2)) # (M,)
  dist_edge=jnp.minimum(dist_edge,jnp.abs(dist_edge3)) # (M,)
  dist_edge=jnp.sqrt(dist_edge**2+dist_surf**2) # (M, N)
  
  msk_isinside=isinside_tri_jx(foot_sruf,tri_msked) # (M,)

  dist=jnp.where(msk_isinside,jnp.abs(dist_surf),dist_edge) # (M,)
  return jnp.abs(dist)

def get_intersection(connect,coord,phi):
  """
  Get the intersection of a triangle mesh and a level set function.

  Parameters
  ----------
  connect : (M, 8) array_like
      The connectivity of the triangle mesh.
  coord : (N, 3) array_like
      The coordinates of the vertices of the triangle mesh.
  phi : (N,) array_like
      The level set function.

  Returns
  -------
  intersection : (M, 3) array_like
      The coordinates of the intersection points.
  """
  edges=connect[:,[[0,1,0,2,0,4]]].reshape(-1,2) #(M*3, 2)
  edges_phi=phi[edges] #(M*3, 2)
  msk_edge=(jnp.sign(edges_phi[:,0])!=jnp.sign(edges_phi[:,1])) # (M*3,)
  edges=edges[msk_edge] # (M*3, 2)
  edges_phi=edges_phi[msk_edge] # (M*3, 2)
  edges_coord=coord[edges] # (M*3, 2, 3)
  edges_phi=jnp.abs(edges_phi)
  coord_intersect=(edges_coord*edges_phi[:,::-1,None]).sum(axis=1)/edges_phi.sum(axis=1,keepdims=True) # (M*3, 3)
  return coord_intersect

def penalty_dist(phi,weightrbf,connect_ls_str,coord_ls_str,nid_const,norm,tri,min_thickness=0.5,max_phi=-1e-5):
  _phi=weightrbf@phi
  v=get_intersection(connect_ls_str,coord_ls_str,_phi)
  
  min_dist=min_distance_to_trimesh(v,norm,tri)

  diff=(min_dist-min_thickness)
  #term1=((diff[diff>0.])**2).mean()*1e3
  #term2=(diff[diff<0.]**2).mean()*1e3
  term12=(diff**2).mean()*1e3
  _phi_const=_phi[nid_const]
  term3=((_phi_const[_phi_const>max_phi]-max_phi)**2).sum()*10
  term4=((phi[nid_const])**2).mean()*1e-5
  #term2=jnp.linalg.norm(jnp.clip(_phi[nid_const],min=max_phi)-max_phi)
  #penalty=term1+term2+term3+term4
  penalty=term12+term3+term4
  #print(f"{stop_gradient(term1):.6f} {stop_gradient(term2):.6f} {stop_gradient(term3):.6f} {stop_gradient(term4):.6f} {stop_gradient(penalty):6f} {stop_gradient(min_dist.min()):.6f} {stop_gradient(min_dist.max()):.6f}")
  print(f"{stop_gradient(term12):.6f} {stop_gradient(term3):.6f} {stop_gradient(term4):.6f} {stop_gradient(penalty):6f} {stop_gradient(min_dist.min()):.6f} {stop_gradient(min_dist.max()):.6f}")
  return penalty

