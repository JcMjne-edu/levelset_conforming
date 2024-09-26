import jax.numpy as jnp

_nid_tetra=jnp.array([[8,21,20,14],[0,8,20,14],[0,9,8,14],[0,13,14,20],[0,8,12,20],
            [8,16,21,14],[1,16,8,14],[1,15,16,14],[1,10,8,16],[1,8,9,14],
            [8,18,21,16],[8,2,18,16],[8,10,2,16],[8,2,11,18],[2,17,18,16],
            [8,21,18,20],[3,8,18,20],[8,11,3,18],[3,12,8,20],[3,18,19,20],
            [14,21,20,26],[14,26,20,4],[4,22,14,26],[13,14,20,4],[20,26,25,4],
            [14,16,21,26],[14,16,26,5],[14,15,16,5],[14,22,5,26],[5,23,16,26],
            [21,16,18,26],[16,6,18,26],[16,17,18,6],[16,23,6,26],[24,18,6,26],
            [21,18,20,26],[20,18,7,26],[20,18,19,7],[18,24,7,26],[25,26,20,7]]) #(40,4)

def redivision(phi,connect):
  """
  phi : (n,)
  connect : (m,8)
  coord : (n,3)
  """
  phis=phi[connect] # (m,8)
  p8=phis[:,[0,1,2,3]].mean(axis=1,keepdims=True) # (m,1)
  p9=phis[:,[0,1]].mean(axis=1,keepdims=True) # (m,1)
  p10=phis[:,[1,2]].mean(axis=1,keepdims=True) # (m,1)
  p11=phis[:,[2,3]].mean(axis=1,keepdims=True) # (m,1)
  p12=phis[:,[0,3]].mean(axis=1,keepdims=True) # (m,1)
  p13=phis[:,[0,4]].mean(axis=1,keepdims=True) # (m,1)
  p14=phis[:,[0,1,4,5]].mean(axis=1,keepdims=True) # (m,1)
  p15=phis[:,[1,5]].mean(axis=1,keepdims=True) # (m,1)
  p16=phis[:,[1,2,5,6]].mean(axis=1,keepdims=True) # (m,1)
  p17=phis[:,[2,6]].mean(axis=1,keepdims=True) # (m,1)
  p18=phis[:,[2,3,6,7]].mean(axis=1,keepdims=True) # (m,1)
  p19=phis[:,[3,7]].mean(axis=1,keepdims=True) # (m,1)
  p20=phis[:,[0,3,4,7]].mean(axis=1,keepdims=True) # (m,1)
  p21=phis.mean(axis=1,keepdims=True) # (m,1)
  p22=phis[:,[4,5]].mean(axis=1,keepdims=True) # (m,1)
  p23=phis[:,[5,6]].mean(axis=1,keepdims=True) # (m,1)
  p24=phis[:,[6,7]].mean(axis=1,keepdims=True) # (m,1)
  p25=phis[:,[4,7]].mean(axis=1,keepdims=True) # (m,1)
  p26=phis[:,[4,5,7,6]].mean(axis=1,keepdims=True) # (m,1)

  phis_ex=jnp.concatenate([phis,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26],axis=1) # (m,27)
  phi_ex=phis_ex.flatten()
  return phi_ex

def redivision_connect_coord(connect,coord):
  """
  phi : (n,)
  connect : (m,8)
  coord : (n,3)
  """
  vertices=coord[connect] # (m,8,3)
  
  v8=vertices[:,[0,1,2,3]].mean(axis=1,keepdims=True) # (m,1,3)
  v9=vertices[:,[0,1]].mean(axis=1,keepdims=True) # (m,1,3)
  v10=vertices[:,[1,2]].mean(axis=1,keepdims=True) # (m,1,3)
  v11=vertices[:,[2,3]].mean(axis=1,keepdims=True) # (m,1,3)
  v12=vertices[:,[0,3]].mean(axis=1,keepdims=True) # (m,1,3)
  v13=vertices[:,[0,4]].mean(axis=1,keepdims=True) # (m,1,3)
  v14=vertices[:,[0,1,4,5]].mean(axis=1,keepdims=True) # (m,1,3)
  v15=vertices[:,[1,5]].mean(axis=1,keepdims=True) # (m,1,3)
  v16=vertices[:,[1,2,5,6]].mean(axis=1,keepdims=True) # (m,1,3)
  v17=vertices[:,[2,6]].mean(axis=1,keepdims=True) # (m,1,3)
  v18=vertices[:,[2,3,6,7]].mean(axis=1,keepdims=True) # (m,1,3)
  v19=vertices[:,[3,7]].mean(axis=1,keepdims=True) # (m,1,3)
  v20=vertices[:,[0,3,4,7]].mean(axis=1,keepdims=True) # (m,1,3)
  v21=vertices.mean(axis=1,keepdims=True) # (m,1,3)
  v22=vertices[:,[4,5]].mean(axis=1,keepdims=True) # (m,1,3)
  v23=vertices[:,[5,6]].mean(axis=1,keepdims=True) # (m,1,3)
  v24=vertices[:,[6,7]].mean(axis=1,keepdims=True) # (m,1,3)
  v25=vertices[:,[4,7]].mean(axis=1,keepdims=True) # (m,1,3)
  v26=vertices[:,[4,5,7,6]].mean(axis=1,keepdims=True) # (m,1,3)

  v_ex=jnp.concatenate([vertices,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26],axis=1) # (m,27,3)
  coord_ex=v_ex.reshape(-1,3)
  connect_ex=jnp.arange(v_ex.shape[0]*v_ex.shape[1]).reshape((v_ex.shape[0],v_ex.shape[1])) # (m,27)
  connect_ex=connect_ex[:,_nid_tetra].reshape(-1,4) # (~,4)
  return connect_ex,coord_ex
