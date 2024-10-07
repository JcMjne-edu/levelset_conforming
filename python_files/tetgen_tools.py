import numpy as np

def make_poly(fname,coord,connect,hole=None):
  """
  Create 

  coord : np.ndarray (n,3)
  connect : np.ndarray (m,3)
  hole : np.ndarray (l,3)
  fname : str
  """
  npoints=coord.shape[0]
  nfaces=connect.shape[0]
  texts=[]
  texts.append(f'{npoints}  3  0  0\n')
  for i,v in enumerate(coord):
    texts.append('{}  {}  {}  {}\n'.format(i+1,*v))
  
  texts.append(f'{nfaces}  0\n')
  for f in connect:
    texts.append('1\n{}  {}  {}  {}\n'.format(3,*f+1))
  if hole is None:
    texts.append('0\n')
  else:
    nholes=hole.shape[0]
    texts.append(f'{nholes}\n')
    for j,h in enumerate(hole):
      texts.append('{}  {}  {}  {}\n'.format(j+1,*h))
  texts.append('0')
  with open(fname,'w') as f:
    f.write(''.join(texts))

def postprocess_tetgen(dir,name,n):
  fname_node=f'{dir}/{name}.{n}.node'
  fname_ele=f'{dir}/{name}.{n}.ele'
  # read node
  f=open(fname_node,'r')
  data=f.read().splitlines()
  f.close()
  nodes=np.array([item.split()[1:] for item in data[1:-1]])
  nodes=nodes.astype(float)
  #read ele
  f=open(fname_ele,'r')
  data=f.read().splitlines()
  f.close()
  ele=np.array([item.split()[1:] for item in data[1:-1]])
  ele=ele.astype(int)-1
  return nodes,ele#,face
