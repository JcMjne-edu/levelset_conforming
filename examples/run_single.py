import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCSR
from jax import config
config.update("jax_enable_x64", True)
from lsto.optimizer import LSOptimizer,get_lr
import numpy as np

k_l=0.03
k_p=2.2
dmul=4.0
young=7e4 # [MPa]
poisson=0.2828
rho=2.7e-9 # [ton/mm^3]

#path to the nastran executable (in this case, Nastran is installed on Windows and run through WSL)
nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'
working_dir='.'

coord_fem=k_l*np.load(f'{working_dir}/param/coord_fem.npy')
coord_geom=k_l*np.load(f'{working_dir}/param/coord_geom.npy')
connect_geom=np.load(f'{working_dir}/param/connect_geom.npy')
nid_trg_rom=np.load(f'{working_dir}/param/nid_rom.npy')
nid_trg_eig=np.load(f'{working_dir}/param/nid_eig.npy')
length_lattice=k_l*np.array([20.0,20.0,20.0])
mass=np.load(f'{working_dir}/param/mass.npy')
compliance_trg=np.load(f'{working_dir}/param/compliance.npy')
eigvec_trg=np.load(f'{working_dir}/param/eigvec_trg.npy')
eigval_trg=np.load(f'{working_dir}/param/eigval_trg.npy')
trgs=(compliance_trg,mass,eigval_trg,eigvec_trg)

#load preprocessed data for the radial basis function smooting matrix
data=jnp.load(f'{working_dir}/param/weightrbf_data.npy')
indices=jnp.load(f'{working_dir}/param/weightrbf_indices.npy')
indptr=jnp.load(f'{working_dir}/param/weightrbf_indptr.npy')
shape=jnp.load(f'{working_dir}/param/weightrbf_shape.npy')
weightrbf=BCSR((data,indices,indptr),shape=shape,indices_sorted=True,unique_indices=True)

#load preprocessed data for the level set mesh
#coord_ls_str=np.load(f'{working_dir}/param/coord_ls_str.npy')
connect_ls_str=np.load(f'{working_dir}/param/connect_ls_str.npy')
nid_const=np.load(f'{working_dir}/param/nid_const.npy')
nid_var=np.load(f'{working_dir}/param/nid_var.npy')
connect_ls_ex=np.load(f'{working_dir}/param/connect_ls_ex.npy')
coord_ls_ex=np.load(f'{working_dir}/param/coord_ls_ex.npy')

lstp=LSOptimizer(connect_ls_str,nid_const,nid_var,connect_ls_ex,coord_ls_ex,coord_geom,
                 connect_geom,weightrbf,length_lattice.min()*0.15)

lstp.set_config(young,poisson,rho,nastran_path,num_mode_trg=6,num_mode_ref=30)
lstp.set_target_raw(k_l,k_p,coord_fem,nid_trg_rom,nid_trg_eig,*trgs)

res_dir='./results/result_20250905/'
phi_boundary=jnp.load(res_dir+'phi_06_40.npy')
scale=1e2
scale2=1e-3
i_offset=sum(
    1 for f in os.listdir(res_dir+'results')
    if os.path.isfile(os.path.join(res_dir+'results', f))
)

phi=jnp.load(res_dir+f'results/phi{i_offset-1}.npy')
grd_func=jax.value_and_grad(lstp.main_eig_static)

lstp.epoch=i_offset-1
loss,grd= grd_func(phi)
lr=get_lr(grd,loss,lstp.nid_var,phi,0.05)
delta=-grd*lr
phi=phi+delta
phi=jnp.clip(phi,-1.,1.)
phi=phi.at[lstp.nid_const].set(phi_boundary[lstp.nid_const])
np.save(res_dir+f'results/phi{i_offset}.npy',np.array(phi))