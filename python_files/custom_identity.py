from jax import custom_vjp
import jax.numpy as jnp

@custom_vjp
def custom_identity(v1,v2):
  return v2

def custom_identity_fwd(v1,v2):
  return v2,None

def custom_identity_bwd(res,g):
  #print('identity grad : ',jnp.abs(g).max())
  return g,None

custom_identity.defvjp(custom_identity_fwd,custom_identity_bwd)