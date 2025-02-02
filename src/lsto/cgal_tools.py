import numpy as np

def read_off(fname):
  """
  Read a mesh from an OFF file.

  Returns:
    vert (n, 3): vertex coordinates
    face (m, 3): triangle vertex indices
    edge (e, 3): edge vertex coordinates
  """
  with open(fname, 'r') as f:
    #lines = f.readlines()
    lines = f.read().splitlines()

  # Read header
  assert lines[0].strip() == 'OFF'
  n, m, _ = map(int, lines[1].strip().split())
  idx=2
  while len(lines[idx])==0:
    idx+=1
  # Read vertices
  vert = []
  for line in lines[idx:idx+n]:
    vert.append(list(map(float, line.strip().split())))
  vert = np.array(vert)

  # Read faces
  face = []
  for line in lines[idx+n:idx+n+m]:
    face.append(list(map(int, line.strip().split()[1:])))
  face = np.array(face)

  # Calculate edges
  edge=face[:,[0,1,1,2,2,0]].reshape((-1,2))
  edge=np.unique(np.sort(edge, axis=1), axis=0)
  edge=vert[edge]
  nones=np.array([None]*len(edge)*3).reshape(-1,1,3)
  edge=np.concatenate([edge,nones], axis=1).reshape(-1,3)

  return vert, face, edge

def write_off(vert,face,fname):
  """
  Write a mesh to an OFF file.
  """
  n=len(vert)
  m=len(face)
  texts=[]
  texts.append(f'OFF\n{n} {m} 0\n')
  for v in vert:
    texts.append(' '.join(map(str,v))+'\n')
  for f in face:
    texts.append(f'3 {" ".join(map(str,f))}\n')
  with open(fname, 'w') as f:
    f.write(''.join(texts))

def read_mesh(fname):
  """
  read coordinates and connectivity from .mesh file
  """
  with open(fname) as f:
    lines = f.read().splitlines()
  n=int(lines[3].split()[0])
  v=np.array([list(map(float, x.split())) for x in lines[4:n+4]])[:,:3]
  m=int(lines[n+5].split()[0])
  l=int(lines[m+n+7].split()[0])
  c=np.array([list(map(int, x.split())) for x in lines[m+n+8:l+m+n+8]])[:,:4]
  return v,c

def extract_surf(connect):
  """
  Extract surface triangles from tetrahedral mesh.
  connect: (n,4)
  """
  tri=connect[:,[0,1,2,0,1,3,0,2,3,1,2,3]].reshape(-1,3)
  tri,count=np.unique(np.sort(tri,axis=1),return_counts=True,axis=0)
  tri=tri[count==1]
  return tri