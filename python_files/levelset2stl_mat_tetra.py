import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from concurrent.futures import ThreadPoolExecutor

def parallel_gi_function(gi_func, phi, connectivity, indices, rev):
    return gi_func(phi, connectivity, indices, rev)

def mat_phi2face_tetra(phi,connectivity,debug=False):
  """
  Compute mapping from level-set function to Gauss Legendre integration points & weights\n
  phi : (nphi,) array of level-set function values at nodes\n
  connectivity : (ne,4) array of element connectivity\n
  Output
  ------
  output1 : BCOO(n, 4, 3, nphi) Gauss Legendre integration mapping\n
  output2 : BCOO(n, 4, nphi) Gauss Legendre integration mapping\n
  label_elm : (n,) element id label\n
  weight_sgn : (n,) sign of weight
  rev_eid : (ne,) element ids with reversed sign
  """
  phis=phi[connectivity] # (ne,8) array of level-set function values at element nodes
  msk=(phis>0.).sum(axis=-1) # (ne,) array of number of nodes with negative level-set function
  
  with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(parallel_gi_function, gi_func, phi, connectivity, np.where(msk == i)[0],rev)
        for gi_func,i,rev in zip([gi1, gi2, gi1], [1,2,3],[False,False,True])
    ]
    results = [future.result() for future in futures]
  mp1, mp2, mp3, = results
  #print('target')
  #for i,name in zip([1,2,3,4,4,5,6,7],["gi1", "gi2", "gi3",]):
  #  print(name, np.where(msk == i)[0].shape)

  datas=np.concatenate([mp1[0],mp2[0],mp3[0]],axis=0)
  offset=jnp.concatenate([mp1[3],mp2[3],mp3[3]],axis=0)
  nindice=np.array([mp1[3].shape[0],mp2[3].shape[0],mp3[3].shape[0]])
  nindice_cumsum=np.cumsum(nindice) # (8,)
  for mp,ni in zip([mp2,mp3],nindice_cumsum[:-1]):
    mp[1][:,0]+=ni
    mp[2][:,0]+=ni

  indice1=np.concatenate([mp1[1],mp2[1],mp3[1]],axis=0) # (n,4)
  indice2=np.concatenate([mp1[2],mp2[2],mp3[2]],axis=0) # (n,3)
  n=nindice_cumsum[-1]; nphi=phi.shape[0]
  out1=BCOO((datas,indice1),shape=(n,3,4,nphi))
  return out1,BCOO((datas,indice2),shape=(n,3,nphi)),offset
  
_ad_ids=np.array([[0,1,2,3],[1,0,3,2],[2,0,1,3],[3,0,2,1]])

_blank_ind1=np.ones((0,4),int); blank_ind2=np.ones((0,3),int)
_blank_data=np.ones((0,))
_id2_bases=np.array([[[0,0,1,1,2,2]],[[0,0,2,2,1,1]]])

def gi1(phi,connectivity,eids,rev=False,return_bcoo=False):
  """
  phi : (nphi,)\n
  connectivity : (ne, 4)\n
  eids : (ne1,) array of element ids with 1 positive level-set function\n
  Return
  ------
  numerator : BCOO(n, 3, 4, np)\n
  denominator : BCOO(n, 3, np)\n
  offset : int (n,) offset id\n 
  """
  n=eids.shape[0]; nphi=phi.shape[0]
  if n==0:
    if return_bcoo:
      return 0
    return _blank_data,_blank_ind1,blank_ind2,np.array([],int)
  connect=connectivity[eids] # (ne1,4)
  phis=phi[connect] # (ne,4)
  pids=np.arange(phi.shape[0])[connect] # (ne,4)
  nbools=(phis<0.) if rev else (phis>=0.) # (ne,4)
  indices=[]
  datas=[]
  indices_div=[]
  offset=[]
  ne_start=0
  id2_selector=1
  if rev:
    id2_selector=1-id2_selector
  count=0
  for ad_id in _ad_ids:
    id2_base=_id2_bases[id2_selector]
    table=np.where(nbools[:,ad_id[0]])[0] #(ne2,)
    ne2=table.shape[0]
    count+=ne2
    if ne2==0: continue
    eid=eids[table] #(ne2,)
    pid=pids[table][:,ad_id] #(ne2,4)
    id1=np.arange(ne2).repeat(6)+ne_start #(ne2*6,)
    id2=id2_base.repeat(ne2,axis=0).flatten() #(ne2*6,)
    id3_1=np.array(ad_id[0]).repeat(ne2) #(ne2,)
    id3_2=np.array(ad_id[1]).repeat(ne2) #(ne2,)
    id3_3=np.array(ad_id[2]).repeat(ne2) #(ne2,)
    id3_4=np.array(ad_id[3]).repeat(ne2) #(ne2,)
    id3=np.array([id3_1,id3_2,id3_1,id3_3,id3_1,id3_4]).T.flatten() #(ne2*6,)
    id4_0=np.zeros(ne2,int)
    id4_1=np.array(pid[:,0]) #(ne2,)
    id4_2=np.array(pid[:,1]) #(ne2,)
    id4_3=np.array(pid[:,2]) #(ne2,)
    id4_4=np.array(pid[:,3]) #(ne2,)
    id4=np.array([id4_2,id4_1,id4_3,id4_1,id4_4,id4_1]).T.flatten() #(ne2*6,)
    data=np.array([1,-1,1,-1,1,-1]).repeat(ne2).reshape(6,ne2).T.flatten() #(ne2*6,)
    indice1=np.array([id1,id2,id3,id4]).T #(ne2*6,4)
    indice2=np.array([id1,id2,id4]).T #(ne2*6,3)
    indices.append(indice1); datas.append(data)
    indices_div.append(indice2)
    offset.append(eid) #(ne2,)
    ne_start+=ne2
  indice1=np.concatenate(indices,axis=0) #(ne*6,4)
  datas=np.concatenate(datas,axis=0) #(ne*6,)
  indice2=np.concatenate(indices_div,axis=0) #(ne*6,3)
  offset=np.concatenate(offset,axis=0) #(ne,)
  #print(offset)
  if return_bcoo:
    return BCOO((datas,indice1),shape=(n,3,4,nphi)),BCOO((datas,indice2),shape=(n,3,nphi)),offset
  return datas,indice1,indice2,offset

def gi2(phi,connectivity,eids,return_bcoo=False):
  """
  phi : (nphi,)\n
  connectivity : (ne, 4)\n
  eids : (ne1,) array of element ids with 2 positive level-set function\n
  Return
  ------
  numerator : BCOO(n, 3, 4, np)\n
  denominator : BCOO(n, 3, np)\n
  offset : int (n,) offset id\n
  """
  n=eids.shape[0]*2; nphi=phi.shape[0]
  if n==0:
    if return_bcoo:
      return 0
    return _blank_data,_blank_ind1,blank_ind2,np.array([],int)
  nids=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
  nids_rev=[[2,3],[1,3],[1,2],[0,3],[0,2],[0,1]]
  id2_selector=np.array([0,1,0,0,1,0])
  connect=connectivity[eids] # (ne1,4)
  phis=phi[connect] # (ne,4)
  pids=np.arange(phi.shape[0])[connect] # (ne,4)
  nbools=(phis<0.)
  indices=[]
  datas=[]
  indices_div=[]
  offset=[]
  ne_start=0
  count=0
  for nid,nid_rev,id2_select in zip(nids,nids_rev,id2_selector):
    table=np.where(nbools[:,nid].all(axis=1))[0] #(ne2,)
    ne2=table.shape[0]
    if ne2==0: continue
    count+=ne2
    id2_base=_id2_bases[id2_select]
    eid=eids[table] #(ne2,)
    pid_local=pids[table] #(ne2,4)
    id3_1=np.array(nid[0]).repeat(ne2) #(ne2,)
    id3_2=np.array(nid[1]).repeat(ne2) #(ne2,)
    id3_3=np.array(nid_rev[0]).repeat(ne2) #(ne2,)
    id3_4=np.array(nid_rev[1]).repeat(ne2) #(ne2,)
    id4_0=np.zeros(ne2,int)
    id4_1=np.array(pid_local[:,nid[0]]) #(ne2,)
    id4_2=np.array(pid_local[:,nid[1]]) #(ne2,)
    id4_3=np.array(pid_local[:,nid_rev[0]]) #(ne2,)
    id4_4=np.array(pid_local[:,nid_rev[1]]) #(ne2,)

    id1=(np.arange(ne2)*2).repeat(6)+ne_start #(ne2*6,)
    id2=id2_base.repeat(ne2,axis=0).reshape(-1) #(ne2*6,)
    id3=np.array([id3_1,id3_3,id3_1,id3_4,id3_2,id3_4]).T.flatten() #(ne2*6,)
    id4=np.array([id4_3,id4_1,id4_4,id4_1,id4_4,id4_2]).T.flatten() #(ne2*6,)
    data=np.array([1,-1,1,-1,1,-1]).repeat(ne2).reshape(-1,ne2).T.flatten() #(ne2*6,)
    indice1=np.array([id1,id2,id3,id4]).T #(ne2*7,4)
    indice2=np.array([id1,id2,id4]).T #(ne2*7,3)
    indices.append(indice1); indices_div.append(indice2)
    datas.append(data)

    id1=(np.arange(ne2)*2+1).repeat(6)+ne_start #(ne2*6,)
    id3=np.array([id3_2,id3_4,id3_2,id3_3,id3_1,id3_3]).T.flatten() #(ne2*6,)
    id4=np.array([id4_4,id4_2,id4_3,id4_2,id4_3,id4_1]).T.flatten() #(ne2*6,)
    indice1=np.array([id1,id2,id3,id4]).T #(ne2*6,4)
    indice2=np.array([id1,id2,id4]).T #(ne2*6,3)
    indices.append(indice1); indices_div.append(indice2)
    datas.append(data)

    offset.append(eid.repeat(2)) #(ne2*3,)
    ne_start+=ne2*2
  indice1=np.concatenate(indices,axis=0) #(ne*6,4)
  datas=np.concatenate(datas,axis=0) #(ne*6,1)
  indice2=np.concatenate(indices_div,axis=0) #(ne*6,3)
  offset=np.concatenate(offset,axis=0) #(ne,)
  if return_bcoo:
    return BCOO((datas,indice1),shape=(n,3,4,nphi)),BCOO((datas,indice2),shape=(n,3,nphi)),offset
  return datas,indice1,indice2,offset
