from lsto.tetgen_tools import postprocess_tetgen
import numpy as np
from pyNastran.bdf.bdf import BDF, CaseControlDeck
from lsto.wing_box import *
import os

kl=0.03
kt=np.sqrt(1/1283.030826740159)
kcomp=2.895688281739209
kv=kl/kt
kf=kl/kcomp
rho_air_scale=kf/kl**2/kv**2
vmin=1e4
vmax=5e5*kv
nvelocity=40
num_modes=15
nx=10
ny=50
anglea=5.0*np.pi/180.0

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
q=0.5*rho_air*(vmax*0.6)**2*rho_air_scale*kv**2

wing2d=Wing2D(f_name_dat)
wing3d=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,locs_lib,ref_span_elem_scale)

nodes_tet,elems_tet,faces_tet,face_marker,_=postprocess_tetgen('./tetgen','temp_OOD',1)
with open("./tetgen/temp_OOD.poly", "r", encoding="utf-8") as file:
    first_line = file.readline().strip()
nnode_original=int(first_line.split()[0])
c=faces_tet[face_marker==2]
vert=nodes_tet[c]
norm=np.cross(vert[:,1]-vert[:,0],vert[:,2]-vert[:,0])
norm=norm/np.linalg.norm(norm,axis=1,keepdims=True)
msk=(norm[:,2]>0.7)
nid_surf=np.unique(c[msk])
spcid=np.where(nodes_tet[:,1]==0.0)[0]
nid_surf=np.setdiff1d(nid_surf,spcid)
nid_tip=np.where(nodes_tet[:,1]==nodes_tet[:,1].max())[0]
nid_surf=np.setdiff1d(nid_surf,nid_tip)
nid_surf=nid_surf[nid_surf<nnode_original]

crd=nodes_tet[nid_surf]
val=crd[:,1]+0.01*crd[:,0]
idx=np.argsort(val)[::-1]
nid_surf=nid_surf[idx]
nid_surf=np.sort(nid_surf.reshape(59,-1)[:,2::3].flatten())


z_offset=np.mean(nodes_tet[nid_surf][:,2])
bdf=BDF(debug=None)
bdf.sol=145
cc=CaseControlDeck(['ECHO=NONE','METHOD=100','SPC=1','FMETHOD=40','SDAMP=2000'])
bdf.case_control_deck=cc
bdf.add_param('POST',-1)
bdf.add_param('KDAMP',1)
bdf.add_param('LMODES',num_modes)
bdf.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)
bdf.add_tabdmp1(2000,x=[0.0,10.0],y=[0.0,0.0])
bdf.add_aero(None,cref=wing3d.root_chord*kl/2,rho_ref=rho_air*rho_air_scale)
bdf.add_eigrl(100,nd=num_modes,norm='MAX')
bdf.add_mkaero1(machs=[0.0,],reduced_freqs=[0.001,1.0])
bdf.add_flutter(40,method='PK',density=1,mach=2,reduced_freq_velocity=4,imethod='S')
bdf.add_flfact(1,[1.])
bdf.add_flfact(2,[0.0])
bdf.add_flfact(4,np.linspace(vmin,vmax,nvelocity))
p4_x=wing3d.sin_sweep*wing3d.semispan*kl+0.25*wing3d.root_chord*kl-0.25*wing3d.tip_chord*kl
p4_y=wing3d.semispan*kl
bdf.add_caero1(1,1,1,p1=[0.,0.,-z_offset],x12=wing3d.root_chord*kl,
               p4=[p4_x,p4_y,-z_offset],x43=wing3d.tip_chord*kl,nchord=nx,nspan=ny)
bdf.add_paero1(1)
bdf.add_spline1(1,1,1,nx*ny,2,0.03)
bdf.add_set1(2,nid_surf+1)
bdf.add_include_file('./nastran_input/data.bdf')
bdf.add_psolid(1,1)
#bdf.add_spc1(1,'123',spcid+1)
fname145='./nastran_input/bulk_aero145.bdf'
bdf.write_bdf(fname145,write_header=False,interspersed=False)
newline='nastran tetraar=4000.0\n'
with open(fname145, 'r') as file:
  lines = file.readlines()
lines.insert(0, newline)
with open(fname145, 'w') as file:
  file.writelines(lines)

#144

bdf=BDF(debug=None)
bdf.sol=144
cc=CaseControlDeck(['ECHO=NONE','SPC=1','AEROF=ALL','APRES=ALL','DISP=ALL','TRIM=1',])
bdf.case_control_deck=cc
bdf.add_param('POST',-1)
bdf.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)
p4_x=wing3d.sin_sweep*wing3d.semispan*kl+0.25*wing3d.root_chord*kl-0.25*wing3d.tip_chord*kl
p4_y=wing3d.semispan*kl
bdf.add_caero1(1,1,1,p1=[0.,0.,-z_offset],x12=wing3d.root_chord*kl,
               p4=[p4_x,p4_y,-z_offset],x43=wing3d.tip_chord*kl,nchord=nx,nspan=ny)
bdf.add_paero1(1,)
bdf.add_spline1(1,1,1,nx*ny,2,0.03)
bdf.add_set1(2,nid_surf+1)
bdf.add_include_file('./nastran_input/data.bdf')
bdf.add_psolid(1,1)
bdf.add_aestat(501,'ANGLEA')
bdf.add_aestat(502,'PITCH')
area=(wing3d.root_chord+wing3d.tip_chord)*wing3d.semispan/2*kl**2
bdf.add_aeros(wing3d.root_chord*kl*0.5,2.*wing3d.semispan*kl,area,sym_xz=1)
bdf.add_trim(1,0.,q,['ANGLEA','PITCH'],[anglea,0.])
#bdf.add_spc1(1,'123',spcid+1)
fname144='./nastran_input/bulk_aero144.bdf'
bdf.write_bdf(fname144,write_header=False,interspersed=False)
newline='nastran tetraar=4000.0\n'
with open(fname144, 'r') as file:
  lines = file.readlines()
lines.insert(0, newline)
with open(fname144, 'w') as file:
  file.writelines(lines)

nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'
command=nastran_path+f' {fname145} out=./nastran_input news=no old=no'
os.system(command)
print('kf',kf)
print('kv',kv)
print('kl',kl)
print('kt',kt)
print('kr',rho_air_scale)