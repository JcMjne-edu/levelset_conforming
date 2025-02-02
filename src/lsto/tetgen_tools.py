import numpy as np

def make_poly(fname,coord,connect,hole=None,face_marker=None,attribute=False):
  """
  coord : np.ndarray (n,3)
  connect : np.ndarray (m,3)
  hole : np.ndarray (l,3)
  face_marker : np.ndarray (m,)
  fname : str
  """
  with open(fname,'w') as f:
    # Write node information
    npoints=coord.shape[0]
    n=16-int(np.log10(np.abs(coord).max()))
    if attribute:
      f.write(f'{npoints}  3  {1}  0\n')
      nids=np.arange(1,npoints+1)
      np.savetxt(f,np.column_stack((nids,coord,np.ones(npoints,dtype=int))),
                fmt=f'%d %.{n}f %.{n}f %.{n}f %d')
    else:
      f.write(f'{npoints}  3  0  0\n')
      np.savetxt(f,np.column_stack((np.arange(1,npoints+1),coord)),
                fmt=f'%d %.{n}f %.{n}f %.{n}f')
    # Write face information
    nfaces=connect.shape[0]
    if face_marker is None:
      f.write(f'{nfaces}  0\n')
      faces=np.column_stack((np.ones(nfaces,dtype=int)*1,
                             np.ones(nfaces,dtype=int)*3,
                             connect+1))
      np.savetxt(f,faces,fmt='%d\n%d %d %d %d')
    else:
      f.write(f'{nfaces}  1\n')
      faces=np.column_stack((np.ones(nfaces,dtype=int)*1,
                             face_marker,
                             np.ones(nfaces,dtype=int)*3,
                             connect+1))
      np.savetxt(f,faces,fmt='%d 0 %d\n%d %d %d %d')
    # Write hole information
    if hole is None:
      f.write('0\n')
    else:
      nholes=hole.shape[0]
      f.write(f'{nholes}\n')
      hole_data=np.column_stack((np.arange(1,nholes+1),hole))
      np.savetxt(f,hole_data,fmt=f'%d %.{n}f %.{n}f %.{n}f')
    f.write('0')

def read_poly(fname):
  with open(fname) as f:
    lines=f.read().splitlines()
  nnode=int(lines[0].split()[0])
  nodes=[lines[i].split()[1:] for i in range(1,nnode+1)]
  nelems=int(lines[nnode+1].split()[0])
  elems=[lines[i*2+nnode+3].split()[1:] for i in range(nelems)]
  nodes=np.array(nodes,dtype=float)
  elems=np.array(elems,dtype=int)-1
  
  return nodes,elems

def postprocess_tetgen(dir,name,n,attribute=False):
  fname_node=f'{dir}/{name}.{n}.node'
  fname_ele=f'{dir}/{name}.{n}.ele'
  fname_face=f'{dir}/{name}.{n}.face'
  if attribute:
    nodes=np.loadtxt(fname_node,skiprows=1,usecols=(1,2,3,4))
    attribute=nodes[:,-1].astype(int)
    nodes=nodes[:,:-1]
  else:
    nodes=np.loadtxt(fname_node,skiprows=1,usecols=(1,2,3))
    attribute=None
  ele=np.loadtxt(fname_ele,skiprows=1,usecols=(1,2,3,4),dtype=int)-1
  f=np.loadtxt(fname_face,skiprows=1,usecols=(1,2,3,4),dtype=int)
  face=f[:,:3]-1
  face_marker=f[:,3]
  return nodes,ele,face,face_marker,attribute
