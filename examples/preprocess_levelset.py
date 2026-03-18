import jax
import jax.numpy as jnp
import numpy as np
import os
from lsto.calculate_init_phi import get_norm, penalty_dist
from lsto.levelset_redivision_med import redivision_connect_coord
from lsto.mesh_tools import meshbuilder_aug
from lsto.mesh_tools.preprocess import cs_rbf_cKDTree

working_dir="."

f_name_dat='./wing2d_dat/nasasc2-0714.dat'
fname_base='./nastran_input/wing3d_skin_flex'
fname1=fname_base+'.bdf'
fname2=fname_base+'_mat.bdf'
span=8000. #[mm]
root_chord=2000. # [mm]
ref_span_elem_scale=0.02
thickness_root=2.5 # [mm]
thickness_tip=1. # [mm]
young=7e4 # [MPa]
poisson=0.3
rho=2.7e-9 # [ton/mm^3]
rho_air=1.0e-12 # [ton/mm^3]
num_modes=15
locs_spar=[0.15,0.75]
locs_lib=list(np.linspace(0,1,21)[:-1])
wing2d=Wing2D(f_name_dat)
wing3d=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,locs_lib,ref_span_elem_scale)
os.makedirs('./nastran_input',exist_ok=True)
os.makedirs(f'{working_dir}/param',exist_ok=True)
wing3d.export_nastran(fname1,fname2,thickness_root,thickness_tip,young,poisson,rho,num_modes=num_modes)
nnode=wing3d.coordinates.shape[0]


command=nastran_path+f' {fname1} out=./nastran_input news=no old=no'
os.system(command)
command=nastran_path+f' {fname2} out=./nastran_input news=no old=no'
os.system(command)

with open(fname_base+'.f06','r') as f:
  lines=f.read().splitlines()
key='                          MASS AXIS SYSTEM (S)     MASS              X-C.G.        Y-C.G.        Z-C.G.'
idx=lines.index(key)
mass=float(lines[idx+1].split()[1])
np.save(f'{working_dir}/param/mass.npy',mass)

op2model=read_op2(fname_base+'.op2',debug=None)
np.save(f'{working_dir}/param/eigvec_trg.npy',op2model.eigenvectors[1].data[:,:,:3])
np.save(f'{working_dir}/param/eigval_trg.npy',op2model.eigenvectors[1].eigns)
pchmodel=read_bdf(fname_base+'_mat.pch',debug=None,punch=True)
kg=pchmodel.dmig['KAAX'].get_matrix(is_sparse=True)[0].tocsc()
unid,count=np.unique(wing3d.connect_quad,return_counts=True)
val=(wing3d.coordinates@np.array([1,1e2,1e-2]))[unid]
idx=np.argsort(val)
nid_rom=np.sort(unid[idx.reshape(-1,103)[4::4][:,[0,21,48,72,-2]].flatten()])

mapping_nid=np.ones(nnode)*(nnode+1)
nid_free=np.where(wing3d.coordinates[:,1]!=0.0)[0]
mapping_nid[nid_free]=np.arange(len(nid_free))
nid_rom_mapped=mapping_nid[nid_rom]

kg_rom=reduction_from_global(kg,nid_rom_mapped)
compliance=np.linalg.inv(kg_rom)
np.save(f'{working_dir}/param/compliance.npy',compliance)
np.save(f'{working_dir}/param/nid_rom.npy',nid_rom)

np.save(f'{working_dir}/param/coord_geom.npy',wing3d.coordinates_surface)
np.save(f'{working_dir}/param/connect_geom.npy',wing3d.connectivity_surface)
np.save(f'{working_dir}/param/coord_fem.npy',wing3d.coordinates)

wing3d_simple=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,[0,1],1.0)
wing3d_simple.export_nastran(fname1,fname2,thickness_root,thickness_tip,young,poisson,rho,num_modes=num_modes)
coord_geom_simple=wing3d_simple.coordinates_surface
connect_geom_simple=wing3d_simple.connectivity_surface
vert=coord_geom_simple[connect_geom_simple] #(n_elem,3,3)
connect_geom_simple=connect_geom_simple[(vert[:,:,1]>0.0).any(axis=1)]
np.save(f'{working_dir}/param/coord_geom_simple.npy',coord_geom_simple)
np.save(f'{working_dir}/param/connect_geom_simple.npy',connect_geom_simple)

k_l=0.03
dmul=2.4
young=7e4 # [MPa]
poisson=0.3
rho=2.7e-9 # [ton/mm^3]
coord_geom=k_l*np.load(f'{working_dir}/param/coord_geom.npy')
connect_geom=np.load(f'{working_dir}/param/connect_geom.npy')
coord_geom_simple=k_l*np.load(f'{working_dir}/param/coord_geom_simple.npy')
connect_geom_simple=np.load(f'{working_dir}/param/connect_geom_simple.npy')
length_lattice=k_l*np.array([20.0,20.0,20.0])
mass=np.load(f'{working_dir}/param/mass.npy')

coord_ls_str,connect_ls_str,nid_const=meshbuilder_aug.meshbuilder_aug(coord_geom,connect_geom,*length_lattice)
nid_var=np.setdiff1d(np.arange(len(coord_ls_str)),nid_const)
connect_ls_ex,coord_ls_ex=redivision_connect_coord(connect_ls_str,coord_ls_str)

nid_root=np.where(coord_ls_str[:,1]==0.0)[0]
_,count=np.unique(connect_ls_str,return_counts=True)
nid_var_additional=np.intersect1d(np.where(count==4)[0],nid_root)
nid_const=np.setdiff1d(nid_const,nid_var_additional)
nid_var=np.union1d(nid_var,nid_var_additional)
np.save(f'{working_dir}/param/coord_ls_str.npy',coord_ls_str)
np.save(f'{working_dir}/param/connect_ls_str.npy',connect_ls_str)
np.save(f'{working_dir}/param/nid_const.npy',nid_const)
np.save(f'{working_dir}/param/nid_var.npy',nid_var)
np.save(f'{working_dir}/param/connect_ls_ex.npy',connect_ls_ex)
np.save(f'{working_dir}/param/coord_ls_ex.npy',coord_ls_ex)

weightrbf=cs_rbf_cKDTree(coord_ls_str/length_lattice,dmul)
np.save(f'{working_dir}/param/weightrbf_data.npy',np.array(weightrbf.data))
np.save(f'{working_dir}/param/weightrbf_indices.npy',np.array(weightrbf.indices))
np.save(f'{working_dir}/param/weightrbf_indptr.npy',np.array(weightrbf.indptr))
np.save(f'{working_dir}/param/weightrbf_shape.npy',np.array(weightrbf.shape))

tri=jnp.array(coord_geom_simple[connect_geom_simple])
norm=get_norm(connect_geom_simple,coord_geom_simple)
df=jax.value_and_grad(penalty_dist)
ratio=0.03

phi=jnp.ones(coord_ls_str.shape[0])
phi=phi.at[nid_const].set(-30.)

scale=1e-5
max_abs=30.0
phi_list=[]
for i in range(200):
  phi_list.append(phi)
  loss,grd=df(phi,weightrbf,connect_ls_str,coord_ls_str,norm,tri,nid_const,min_thickness=0.5)
  grd=grd.at[nid_var].set(0.0)
  grd=grd.at[jnp.abs(grd)==max_abs].set(0.0)
  alpha=ratio*loss/(grd**2).sum()
  phi=phi - alpha*grd - phi.at[nid_var].set(0.0)*scale
  phi=jnp.clip(phi,-max_abs,1.0)
#np.save(f'{working_dir}/param/init_phi_list.npy',np.array(phi_list))
np.save(f'{working_dir}/param/init_phi.npy',np.array(phi))
