import numpy as np
from stl import mesh

def stl_from_connect_and_coord(connect,coord):
  """
  connect: (nelem,3) array of element connectivity\\
  coord: (nnode,3) array of node coordinates
  """
  connect=np.asarray(connect)
  coord=np.asarray(coord)
  nelem=connect.shape[0]
  mesh_data=mesh.Mesh(np.zeros(nelem,dtype=mesh.Mesh.dtype))
  for i,f in enumerate(connect):
    for j in range(3):
      mesh_data.vectors[i][j]=coord[f[j],:]
  return mesh_data

def stl_from_mesh3d(mesh3d):
  """
  mesh3d: (n,3,3) array of node coordinates
  """
  n=mesh3d.shape[0]
  mesh_data=mesh.Mesh(np.zeros(n,dtype=mesh.Mesh.dtype))
  for i in range(n):
    for j in range(3):
      mesh_data.vectors[i][j]=mesh3d[i,j]
  return mesh_data