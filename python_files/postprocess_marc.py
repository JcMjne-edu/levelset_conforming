import numpy as np

def read_marc_eig_6dof(f_name):
  """
  Return
  -------
  all_modes : (num_mode,num_node)
  all_freqs : (num_mode,) rad/s
  """
  f=open(f_name,'r')
  data=f.read().splitlines()
  f.close()
  key="                              e i g e n v e c t o r"
  si=0
  all_modes=[]
  all_freqs=[]
  while True:
    try:
      si=data[si:].index(key)+si+3
    except ValueError:
      break
    mode_data=[]
    i=0
    while True:
      if data[si+i]=='':
        freq=np.array(data[si-7].split()[6]).astype(float)
        all_freqs.append(freq)
        break
      mode_data.append(np.array(data[si+i].split()[1:4]).astype(float))
      i+=1
    all_modes.append(mode_data)
  return np.array(all_modes),np.array(all_freqs)

def read_marc_static_6dof(f_name):
  """
  Return
  -------
  deflection : (num_node,3)
  """
  f=open(f_name,'r')
  data=f.read().splitlines()
  f.close()
  key="                              t o t a l   d i s p l a c e m e n t s "
  deflection=[]
  si=data.index(key)+3
  i=0
  while True:
    if data[si+i]=='':
      break
    else:
      deflection.append(np.array(data[si+i].split()[1:4]).astype(float))
      i+=1
  return np.array(deflection)

def read_marc_matrix_tet(f_name):
  f=open(f_name,'r')
  data=f.read().splitlines()
  f.close()
  key='                              element stiffness matrix          1'
  id=data.index(key)+4
  tetra_lst=[]

  while True:
    mat_frac=[]
    for j in range(2):
      temp_lst=[]
      for k in range(12):
        temp=data[id].replace('D','E').split()[2:]
        temp_lst.append(np.array(temp).astype(float))
        id+=1
      mat_frac.append(np.array(temp_lst))
      id+=3
    tetra_lst.append(np.concatenate(mat_frac,axis=1))
    if data[id-2]=='                              theta matrix for element 1 and position         1':
      break
    id+=6

  key='                              element mass matrix           1'
  id=data.index(key)+4
  tetra_mlst=[]
  while True:
    mat_frac=[]
    for j in range(2):
      temp_lst=[]
      for k in range(12):
        temp=data[id].replace('D','E').split()[2:]
        temp_lst.append(np.array(temp).astype(float))
        id+=1
      mat_frac.append(np.array(temp_lst))
      id+=3
    tetra_mlst.append(np.diag(np.concatenate(mat_frac,axis=1)))
    id+=5
    if len(data[id-8])==0:
      break
  
  ke=np.array(tetra_lst)
  me=np.array(tetra_mlst)
  return ke,me

def read_marc_eig_3dof(f_name):
  """
  Return
  -------
  all_modes : (num_mode,num_node)
  all_freqs : (num_mode,) rad/s
  """
  f=open(f_name,'r')
  data=f.read().splitlines()
  f.close()
  key="                              e i g e n v e c t o r"
  si=0
  all_modes=[]
  all_freqs=[]
  while True:
    try:
      si=data[si:].index(key)+si+3
    except ValueError:
      break
    mode_data=[]
    i=0
    while True:
      if data[si+i]=='':
        freq=np.array(data[si-7].split()[6]).astype(float)
        all_freqs.append(freq)
        break
      temp=data[si+i].split()
      mode_data.append(np.array(temp[1:4]).astype(float))
      if len(temp)==8:
        mode_data.append(np.array(temp[5:8]).astype(float))
      i+=1
    all_modes.append(mode_data)
  return np.array(all_modes),np.array(all_freqs)

def read_marc_static_3dof(f_name):
  """
  Return
  -------
  deflection : (num_node,3)
  """
  f=open(f_name,'r')
  data=f.read().splitlines()
  f.close()
  key="                              t o t a l   d i s p l a c e m e n t s "
  deflection=[]
  si=data.index(key)+3
  i=0
  while True:
    if data[si+i]=='':
      break
    else:
      temp=data[si+i].split()
      deflection.append(np.array(temp[1:4]).astype(float))
      if len(temp)==8:
        deflection.append(np.array(temp[5:8]).astype(float))
      i+=1
  return np.array(deflection)

def read_marc_ke_mix(f_name):
  f=open(f_name,'r')
  data=f.read().splitlines()
  f.close()
  key='                              element stiffness matrix          1'
  idx=data.index(key)+4
  quad_lst=[]

  while True:
    mat_frac=[]
    for j in range(4):
      temp_lst=[]
      for k in range(24):
        temp=data[idx].replace('D','E').split()[2:]
        temp_lst.append(np.array(temp).astype(float))
        idx+=1
      mat_frac.append(np.array(temp_lst))
      idx+=3
    quad_lst.append(np.concatenate(mat_frac,axis=1))
    if len(data[idx-3])==0 and len(data[idx-2])==0:
      break
    idx+=9
  eid=len(quad_lst)+1
  key=f'                              element stiffness matrix      {eid:>5}'
  idx=data.index(key)+4
  tri_lst=[]

  while True:
    mat_frac=[]
    for j in range(3):
      temp_lst=[]
      for k in range(18):
        temp=data[idx].replace('D','E').split()[2:]
        temp_lst.append(np.array(temp).astype(float))
        idx+=1
      mat_frac.append(np.array(temp_lst))
      idx+=3
    tri_lst.append(np.concatenate(mat_frac,axis=1))
    if len(data[idx])==0 and len(data[idx-2])==0:
      break
    idx+=8

  ke_quad=np.array(quad_lst)
  ke_tri=np.array(tri_lst)
  return ke_tri,ke_quad