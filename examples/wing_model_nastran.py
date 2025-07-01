from lsto.wing_box import *
from lsto.mesh_tools.preprocess import *
from lsto.rom.k_reduction_6dof import reduction_from_global
import os
from pyNastran.op2.op2 import read_op2
from pyNastran.bdf.bdf import read_bdf

f_name_dat='./wing2d_dat/nasasc2-0714.dat'
fname_base='./nastran_input/wing3d_skin_flex'
fname1=fname_base+'.bdf'
fname2=fname_base+'_mat.bdf'
span=8000. #[mm]
root_chord=2000. # [mm]
tip_chord=800. # [mm]
sweep=30.0 # [deg]
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
os.makedirs('./nastran_trg',exist_ok=True)
wing3d.export_nastran(fname1,fname2,thickness_root,thickness_tip,young,poisson,rho,num_modes=num_modes)
nnode=wing3d.coordinates.shape[0]

nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'
command=nastran_path+f' {fname1} out=./nastran_input news=no old=no'
os.system(command)
command=nastran_path+f' {fname2} out=./nastran_input news=no old=no'
os.system(command)

with open(fname_base+'.f06','r') as f:
  lines=f.read().splitlines()
key='                          MASS AXIS SYSTEM (S)     MASS              X-C.G.        Y-C.G.        Z-C.G.'
idx=lines.index(key)
mass=float(lines[idx+1].split()[1])
np.save('./nastran_trg/mass.npy',mass)

op2model=read_op2(fname_base+'.op2',debug=None)
np.save('./nastran_trg/eigvec_trg.npy',op2model.eigenvectors[1].data[:,:,:3])
np.save('./nastran_trg/eigval_trg.npy',op2model.eigenvectors[1].eigns)

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
np.save('./nastran_trg/compliance.npy',compliance)
np.save('./nastran_trg/nid_rom.npy',nid_rom)

np.save('./nastran_trg/coord_geom.npy',wing3d.coordinates_surface)
np.save('./nastran_trg/connect_geom.npy',wing3d.connectivity_surface)
np.save('./nastran_trg/coord_fem.npy',wing3d.coordinates)