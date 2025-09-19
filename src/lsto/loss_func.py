import jax.numpy as jnp

def loss_cossim(v1,v2):
  """
  v1 : (n,k1)
  v2 : (n,k1)
  """
  upper=jnp.einsum('ij,ij->j',v1,v2) #(k1,)
  lower=jnp.linalg.norm(v1,axis=0)*jnp.linalg.norm(v2,axis=0) #(k1,)
  weight=1/jnp.arange(1,v1.shape[1]+1)**0.5
  weight=weight/weight.sum()
  out=((1-jnp.abs(upper/lower))**2)@weight
  #out=((1-jnp.abs(upper/lower))**2).mean()
  if lower.min()==0.0:
    raise ValueError("Zero vector detected")
  return out
