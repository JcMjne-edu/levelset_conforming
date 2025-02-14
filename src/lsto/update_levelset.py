import jax.numpy as jnp
import numpy as np

def get_adjecent_nid(connects_ls_str,coords_ls_str):
  adj=np.arange(coords_ls_str.shape[0]).repeat(6).reshape(-1,6)
  #adj=-np.ones((coords_ls_str.shape[0],6),int)
  edge_x=connects_ls_str[:,[0,1,3,2,4,5,7,6]].reshape(-1,2)
  adj[edge_x[:,0],1]=edge_x[:,1]
  adj[edge_x[:,1],0]=edge_x[:,0]

  edge_y=connects_ls_str[:,[0,3,4,7,1,2,5,6]].reshape(-1,2)
  adj[edge_y[:,0],3]=edge_y[:,1]
  adj[edge_y[:,1],2]=edge_y[:,0]

  edge_z=connects_ls_str[:,[0,4,1,5,2,6,3,7]].reshape(-1,2)
  adj[edge_z[:,0],5]=edge_z[:,1]
  adj[edge_z[:,1],4]=edge_z[:,0]
  return jnp.array(adj)

def update_phi(phi,grd,adj,length_lattice,cfl=0.9):
  """
  phi : (nv,)
  grd : (nv,)
  adj : (nv,6)
  """
  op_divergence=jnp.array([-1.,1.,-1.,1.,-1.,1.])/length_lattice.repeat(2)/2.
  lmin=length_lattice.min()
  divergence=phi[adj]@op_divergence #(nv,)
  dT=cfl*lmin/jnp.abs(grd).max()
  return phi-dT*jnp.abs(divergence)*grd