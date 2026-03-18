import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["XLA_FLAGS"] = "--xla_dump_to=stdout --xla_dump_hlo_as_text"
from jax import config
config.update("jax_enable_x64", True)
from lsto.calculate_init_phi import penalty_dist,get_norm
from lsto.wing_box import *
from lsto.mesh_tools.preprocess import *
from lsto.rom.k_reduction_6dof import reduction_from_global
from pyNastran.op2.op2 import read_op2
from pyNastran.bdf.bdf import read_bdf
import jax
from lsto.levelset_penetrate_aug import *
import shutil

def nastran_aero_trg(tag='030_30',tip_chord=600.,sweep=30.0,thickness_root=2.5,thickness_tip=1.,nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'):
  f_name_dat='./wing2d_dat/nasasc2-0714.dat'
  fname_145=f'./case_study/{tag}/wing3d_skin_145.bdf'
  fname_144=f'./case_study/{tag}/wing3d_skin_144.bdf'
  nx=10
  ny=50
  vmin=1e4
  vmax=6e5
  nvelocity=40
  rho_air=1.0e-12 # [ton/mm^3]
  q=0.5*rho_air*(vmax*0.6)**2
  anglea=5.0*np.pi/180.0
  span=8000. #[mm]
  root_chord=2000. # [mm]
  ref_span_elem_scale=0.02
  young=7e4 # [MPa]
  poisson=0.3
  rho=2.7e-9 # [ton/mm^3]
  num_modes=15
  locs_spar=[0.15,0.75]
  locs_lib=list(np.linspace(0,1,21)[:-1])
  wing2d=Wing2D(f_name_dat)
  wing3d=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,locs_lib,ref_span_elem_scale)
  os.makedirs(f'./case_study/{tag}',exist_ok=True)
  wing3d.export_nastran_aero(fname_145,thickness_root,thickness_tip,nx,ny,vmin,vmax,young,poisson,rho,rho_air,num_modes,nvelocity)
  wing3d.export_nastran_aero(fname_144,thickness_root,thickness_tip,nx,ny,vmin,vmax,young,poisson,rho,flutter=False,q=q,anglea=anglea)
  os.system(nastran_path+f' {fname_144} out=./case_study/{tag} news=no old=no')
  os.system(nastran_path+f' {fname_145} out=./case_study/{tag} news=no old=no')

def nastran_aero_bulk(kcomp,kt2,tag='030_30',tip_chord=600.,sweep=30.0,nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'):
  kl=0.03
  kt=np.sqrt(kt2)
  young=7e4 # [MPa]
  poisson=0.2828
  rho=2.7e-9 # [ton/mm^3]
  #----------------
  kv=kl/kt
  kf=kl/kcomp
  rho_air_scale=kf/kl**2/kv**2
  vmin=1e4
  vmax=6e5*kv
  nvelocity=40
  num_modes=15
  nx=10
  ny=50
  anglea=5.0*np.pi/180.0

  f_name_dat='./wing2d_dat/nasasc2-0714.dat'
  span=8000. #[mm]
  root_chord=2000. # [mm]
  ref_span_elem_scale=0.02
  rho_air=1.0e-12 # [ton/mm^3]
  num_modes=15
  locs_spar=[0.15,0.75]
  locs_lib=list(np.linspace(0,1,21)[:-1])
  q=0.5*rho_air*(vmax*0.6)**2*rho_air_scale*kv**2

  wing2d=Wing2D(f_name_dat)
  wing3d=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,locs_lib,ref_span_elem_scale)

  nodes_tet,elems_tet,faces_tet,face_marker,_=postprocess_tetgen(f'./case_study/{tag}','tetgen',1)
  with open(f"./case_study/{tag}/tetgen.poly", "r", encoding="utf-8") as file:
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
  nid_surf=np.sort(nid_surf.reshape(59,-1)[:,2::2].flatten())

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
  bdf.add_include_file(f'./data.bdf') #relative path
  bdf.add_psolid(1,1)
  #bdf.add_spc1(1,'123',spcid+1)
  fname145=f'./case_study/{tag}/bulk_aero145.bdf'
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
  bdf.add_include_file('./data.bdf')
  bdf.add_psolid(1,1)
  bdf.add_aestat(501,'ANGLEA')
  bdf.add_aestat(502,'PITCH')
  area=(wing3d.root_chord+wing3d.tip_chord)*wing3d.semispan/2*kl**2
  bdf.add_aeros(wing3d.root_chord*kl*0.5,2.*wing3d.semispan*kl,area,sym_xz=1)
  bdf.add_trim(1,0.,q,['ANGLEA','PITCH'],[anglea,0.])
  #bdf.add_spc1(1,'123',spcid+1)
  fname144=f'./case_study/{tag}/bulk_aero144.bdf'
  bdf.write_bdf(fname144,write_header=False,interspersed=False)
  newline='nastran tetraar=4000.0\n'
  with open(fname144, 'r') as file:
    lines = file.readlines()
  lines.insert(0, newline)
  with open(fname144, 'w') as file:
    file.writelines(lines)#

  os.system(nastran_path+f' {fname145} out=./case_study/{tag} news=no old=no')
  os.system(nastran_path+f' {fname144} out=./case_study/{tag} news=no old=no')

def main(tag='030_30',tip_chord=600.,sweep=30.0,thickness_root=2.5,thickness_tip=1.,
         nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'):
  f_name_dat='./wing2d_dat/nasasc2-0714.dat'
  fname_base='./nastran_input/wing3d_skin_flex'
  fname1=fname_base+'.bdf'
  fname2=fname_base+'_mat.bdf'
  span=8000. #[mm]
  root_chord=2000. # [mm]
  ref_span_elem_scale=0.02
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
  os.makedirs(f'./case_study/{tag}',exist_ok=True)
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
  np.save(f'./case_study/{tag}/mass.npy',mass)
  
  op2model=read_op2(fname_base+'.op2',debug=None)
  np.save(f'./case_study/{tag}/eigvec_trg.npy',op2model.eigenvectors[1].data[:,:,:3])
  np.save(f'./case_study/{tag}/eigval_trg.npy',op2model.eigenvectors[1].eigns)
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
  np.save(f'./case_study/{tag}/compliance.npy',compliance)
  np.save(f'./case_study/{tag}/nid_rom.npy',nid_rom)
  
  np.save(f'./case_study/{tag}/coord_geom.npy',wing3d.coordinates_surface)
  np.save(f'./case_study/{tag}/connect_geom.npy',wing3d.connectivity_surface)
  np.save(f'./case_study/{tag}/coord_fem.npy',wing3d.coordinates)
  
  wing3d_simple=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,[0,1],1.0)
  wing3d_simple.export_nastran(fname1,fname2,thickness_root,thickness_tip,young,poisson,rho,num_modes=num_modes)
  coord_geom_simple=wing3d_simple.coordinates_surface
  connect_geom_simple=wing3d_simple.connectivity_surface
  vert=coord_geom_simple[connect_geom_simple] #(n_elem,3,3)
  connect_geom_simple=connect_geom_simple[(vert[:,:,1]>0.0).any(axis=1)]
  np.save(f'./case_study/{tag}/coord_geom_simple.npy',coord_geom_simple)
  np.save(f'./case_study/{tag}/connect_geom_simple.npy',connect_geom_simple)
  
  k_l=0.03
  dmul=2.4
  young=7e4 # [MPa]
  poisson=0.3
  rho=2.7e-9 # [ton/mm^3]
  coord_geom=k_l*np.load(f'./case_study/{tag}/coord_geom.npy')
  connect_geom=np.load(f'./case_study/{tag}/connect_geom.npy')
  coord_geom_simple=k_l*np.load(f'./case_study/{tag}/coord_geom_simple.npy')
  connect_geom_simple=np.load(f'./case_study/{tag}/connect_geom_simple.npy')
  length_lattice=k_l*np.array([20.0,20.0,20.0])
  mass=np.load(f'./case_study/{tag}/mass.npy')

  lstp=LSTP_penetrate_aug(coord_geom,connect_geom,length_lattice,dmul,coord_geom,connect_geom,length_lattice.min()*0.15)
  weightrbf=lstp.weightrbf
  np.save(f'./case_study/{tag}/weightrbf_data.npy',np.array(weightrbf.data))
  np.save(f'./case_study/{tag}/weightrbf_indices.npy',np.array(weightrbf.indices))
  np.save(f'./case_study/{tag}/weightrbf_indptr.npy',np.array(weightrbf.indptr))
  np.save(f'./case_study/{tag}/weightrbf_shape.npy',np.array(weightrbf.shape))

  nid_const=lstp.nid_const
  nid_var=lstp.nid_var
  connect_ls_str=lstp.connect_ls_str
  coord_ls_str=lstp.coord_ls_str

  tri=jnp.array(coord_geom_simple[connect_geom_simple])
  norm=get_norm(connect_geom_simple,coord_geom_simple)
  df=jax.value_and_grad(penalty_dist)
  ratio=0.03
  max_abs=80.0

  phi=jnp.ones(lstp.coord_ls_str.shape[0])
  phi=phi.at[lstp.nid_const].set(-max_abs)

  scale=1e-5
  phi_list=[]
  for i in range(300):
    phi_list.append(phi)
    loss,grd=df(phi,weightrbf,connect_ls_str,coord_ls_str,norm,tri,nid_const,min_thickness=0.8,max_phi=max_abs)
    grd=grd.at[nid_var].set(0.0)
    grd=grd.at[jnp.abs(grd)==max_abs].set(0.0)
    alpha=ratio*loss/(grd**2).sum()
    phi=phi - alpha*grd - phi.at[nid_var].set(0.0)*scale
    phi=jnp.clip(phi,-max_abs,1.0)
    np.save(f'./case_study/{tag}/init_phi_list.npy',np.array(phi_list))
    np.save(f'./case_study/{tag}/init_phi.npy',np.array(phi))
    np.save(f'./case_study/{tag}/results/phi0.npy',np.array(phi))

def run_single(tag='030_30'):
  k_l=0.03
  k_p=2.2
  dmul=4.0
  young=7e4 # [MPa]
  poisson=0.2828
  rho=2.7e-9 # [ton/mm^3]
  nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'
  coord_fem=k_l*np.load(f'./case_study/{tag}/coord_fem.npy')
  coord_geom=k_l*np.load(f'./case_study/{tag}/coord_geom.npy')
  connect_geom=np.load(f'./case_study/{tag}/connect_geom.npy')
  nid_trg_rom=np.load(f'./case_study/{tag}/nid_rom.npy')
  length_lattice=k_l*np.array([20.0,20.0,20.0])
  mass=np.load(f'./case_study/{tag}/mass.npy')
  compliance_trg=np.load(f'./case_study/{tag}/compliance.npy')
  eigvec_trg=np.load(f'./case_study/{tag}/eigvec_trg.npy')
  eigval_trg=np.load(f'./case_study/{tag}/eigval_trg.npy')
  trgs=(compliance_trg,mass,eigval_trg,eigvec_trg)

  data=jnp.load(f'./case_study/{tag}/weightrbf_data.npy')
  indices=jnp.load(f'./case_study/{tag}/weightrbf_indices.npy')
  indptr=jnp.load(f'./case_study/{tag}/weightrbf_indptr.npy')
  shape=jnp.load(f'./case_study/{tag}/weightrbf_shape.npy')
  weightrbf=BCSR((data,indices,indptr),shape=shape,indices_sorted=True,unique_indices=True)

  lstp=LSTP_penetrate_aug(coord_geom,connect_geom,length_lattice,dmul,coord_geom,connect_geom,length_lattice.min()*0.14,weightrbf=weightrbf)
  lstp.set_config(young,poisson,rho,nastran_path,num_mode_trg=6,num_mode_ref=15,mass_scale=1.)
  lstp.set_target_raw(k_l,k_p,coord_fem,nid_trg_rom,nid_trg_rom,*trgs)

  res_dir=f'./case_study/{tag}/results/'
  os.makedirs(res_dir,exist_ok=True)
  phi_boundary=jnp.load(f'./case_study/{tag}/init_phi.npy')
  phi0=phi_boundary
  for i in range(10):
    phi0=lstp.weightrbf@phi0
    phi0=phi0.at[lstp.nid_const].set(phi_boundary[lstp.nid_const])
  np.save(res_dir+'phi0.npy',phi0)
  
  i_offset=sum(
      1 for f in os.listdir(res_dir)
      if os.path.isfile(os.path.join(res_dir, f))
  )
  phi=jnp.load(res_dir+f'phi{i_offset-1}.npy')
  phi=jnp.clip(phi,-1.,1.)
  phi=phi.at[lstp.nid_const].set(phi_boundary[lstp.nid_const])
  grd_func=jax.value_and_grad(lstp.main_eig_static)
  lstp.epoch=i_offset-1
  try:
    loss,grd= grd_func(phi)
    if jnp.isnan(grd).sum():
      raise ValueError("nan detected")
    lr=get_lr_mod(grd,loss,lstp.nid_var,0.05)
    delta=-grd*lr
    phi=phi+delta
    phi=jnp.clip(phi,-1.,1.)
    phi=phi.at[lstp.nid_const].set(phi_boundary[lstp.nid_const])
    np.save(res_dir+f'phi{i_offset}.npy',np.array(phi))
  
  except ValueError as e:
    print(e," at epoch ",i_offset)
    phi_prev=jnp.load(res_dir+f'phi{i_offset-2}.npy')
    delta=phi-phi_prev
    phi=phi+delta*0.9
    phi=jnp.clip(phi,-1.,1.)
    phi=phi.at[lstp.nid_const].set(phi_boundary[lstp.nid_const])
    np.save(res_dir+f'phi{i_offset}.npy',np.array(phi))

def run_forward(tag='030_30',nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'):
  k_l=0.03
  k_p=2.2
  dmul=4.0
  young=7e4 # [MPa]
  poisson=0.2828
  rho=2.7e-9 # [ton/mm^3]
  
  coord_fem=k_l*np.load(f'./case_study/{tag}/coord_fem.npy')
  coord_geom=k_l*np.load(f'./case_study/{tag}/coord_geom.npy')
  connect_geom=np.load(f'./case_study/{tag}/connect_geom.npy')
  nid_trg_rom=np.load(f'./case_study/{tag}/nid_rom.npy')
  length_lattice=k_l*np.array([20.0,20.0,20.0])
  mass=np.load(f'./case_study/{tag}/mass.npy')
  compliance_trg=np.load(f'./case_study/{tag}/compliance.npy')
  eigvec_trg=np.load(f'./case_study/{tag}/eigvec_trg.npy')
  eigval_trg=np.load(f'./case_study/{tag}/eigval_trg.npy')
  trgs=(compliance_trg,mass,eigval_trg,eigvec_trg)

  data=jnp.load(f'./case_study/{tag}/weightrbf_data.npy')
  indices=jnp.load(f'./case_study/{tag}/weightrbf_indices.npy')
  indptr=jnp.load(f'./case_study/{tag}/weightrbf_indptr.npy')
  shape=jnp.load(f'./case_study/{tag}/weightrbf_shape.npy')
  weightrbf=BCSR((data,indices,indptr),shape=shape,indices_sorted=True,unique_indices=True)

  lstp=LSTP_penetrate_aug(coord_geom,connect_geom,length_lattice,dmul,coord_geom,connect_geom,length_lattice.min()*0.14,weightrbf=weightrbf)
  lstp.set_config(young,poisson,rho,nastran_path,num_mode_trg=6,num_mode_ref=15,mass_scale=1.)
  lstp.set_target_raw(k_l,k_p,coord_fem,nid_trg_rom,nid_trg_rom,*trgs)

  res_dir=f'./case_study/{tag}/results/'
  
  i_offset=sum(
      1 for f in os.listdir(res_dir)
      if os.path.isfile(os.path.join(res_dir, f))
  )
  phi=jnp.load(res_dir+f'phi{i_offset-1}.npy')

  lstp.main_eig_static(phi)
  shutil.copyfile("./nastran/data.bdf",f'./case_study/{tag}/data.bdf')
  shutil.copyfile('./tetgen/temp_OOD.poly',f'./case_study/{tag}/tetgen.poly')
  shutil.copyfile('./tetgen/temp_OOD.1.node',f'./case_study/{tag}/tetgen.1.node')
  shutil.copyfile('./tetgen/temp_OOD.1.ele',f'./case_study/{tag}/tetgen.1.ele')
  shutil.copyfile('./tetgen/temp_OOD.1.face',f'./case_study/{tag}/tetgen.1.face')

def visualize_structure(tip_chord=600.,sweep=30.0):
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
  num_modes=15
  locs_spar=[0.15,0.75]
  locs_lib=list(np.linspace(0,1,21)[:-1])
  wing2d=Wing2D(f_name_dat)
  wing3d=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,locs_lib,ref_span_elem_scale)
  wing3d.export_nastran(fname1,fname2,thickness_root,thickness_tip,young,poisson,rho,num_modes=num_modes)
  
  fig=wing3d.to_plotly()
  camera = dict(
    eye=dict(x=0., y=-0., z=2.),
    center=dict(x=0.0, y=-0.0, z=-0.0),   
    up=dict(x=-1, y=0, z=0),
    projection=dict(type='orthographic')
  )
  fig.update_layout(scene=dict(aspectmode='data', 
                              xaxis=dict(visible=False),
                              yaxis=dict(visible=False),
                              zaxis=dict(visible=False),camera=camera),
                    margin=dict(l=0,r=0,b=0,t=0),
                    )
  return fig

def visualize_surface(tag):
  k_l=0.03
  k_p=2.2
  dmul=4.0
  young=7e4 # [MPa]
  poisson=0.2828
  rho=2.7e-9 # [ton/mm^3]
  nastran_path='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'
  coord_fem=k_l*np.load(f'./case_study/{tag}/coord_fem.npy')
  coord_geom=k_l*np.load(f'./case_study/{tag}/coord_geom.npy')
  connect_geom=np.load(f'./case_study/{tag}/connect_geom.npy')
  nid_trg_rom=np.load(f'./case_study/{tag}/nid_rom.npy')
  length_lattice=k_l*np.array([20.0,20.0,20.0])
  mass=np.load(f'./case_study/{tag}/mass.npy')
  compliance_trg=np.load(f'./case_study/{tag}/compliance.npy')
  eigvec_trg=np.load(f'./case_study/{tag}/eigvec_trg.npy')
  eigval_trg=np.load(f'./case_study/{tag}/eigval_trg.npy')
  trgs=(compliance_trg,mass,eigval_trg,eigvec_trg)

  data=jnp.load(f'./case_study/{tag}/weightrbf_data.npy')
  indices=jnp.load(f'./case_study/{tag}/weightrbf_indices.npy')
  indptr=jnp.load(f'./case_study/{tag}/weightrbf_indptr.npy')
  shape=jnp.load(f'./case_study/{tag}/weightrbf_shape.npy')
  weightrbf=BCSR((data,indices,indptr),shape=shape,indices_sorted=True,unique_indices=True)

  lstp=LSTP_penetrate_aug(coord_geom,connect_geom,length_lattice,dmul,coord_geom,connect_geom,length_lattice.min()*0.14,weightrbf=weightrbf)
  lstp.set_config(young,poisson,rho,nastran_path,num_mode_trg=6,num_mode_ref=15,mass_scale=1.)
  lstp.set_target_raw(k_l,k_p,coord_fem,nid_trg_rom,nid_trg_rom,*trgs)

  res_dir=f'./case_study/{tag}/results/'
  i_offset=sum(
      1 for f in os.listdir(res_dir)
      if os.path.isfile(os.path.join(res_dir, f))
  )
  phi=jnp.load(res_dir+f'phi{i_offset-1}.npy')
  
  _phi=lstp.weightrbf@phi
  _phi=_phi.at[lstp.nid_const].set(jnp.clip(_phi[lstp.nid_const],max=-0.1,))
  _phi_ex=redivision(_phi,lstp.connect_ls_str)
  numerator,denominator,offset=mat_phi2face_tetra(_nparray(_phi_ex),lstp.connect_ls_ex)
  mesh3d=((numerator@_phi_ex)@lstp.vertices_ex[offset])/(denominator@_phi_ex)[:,:,None]
  lstp.logger_aux.info(f'mesh3d generated ({mesh3d.shape})')
  for rf in [9,8,7,6,5,4]:
    try:
      coord_index,connect_ls=mesh3d_to_coord_and_connect(_nparray(mesh3d),round_f=rf)
      coord_ls=mesh3d.reshape(-1,3)[coord_index]
      detect_hole_open(connect_ls,coord_ls)
      break
    except ValueError:
      if rf==4:
        raise ValueError('Hole detected!!')
      continue
  
  connect_ls,coord_ls=elim_closed_surface(connect_ls,coord_ls)
  
  fig=go.Figure()
  v=coord_geom
  c=connect_geom
  fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=c[:,0],j=c[:,1],k=c[:,2],opacity=0.5,showscale=False,name='geom'))
  v=np.array(coord_ls)
  c=np.array(connect_ls)
  fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=c[:,0],j=c[:,1],k=c[:,2],opacity=0.5,showscale=False,name='init'))
  camera = dict(
    eye=dict(x=0., y=-0., z=2.),
    center=dict(x=0.0, y=-0.0, z=-0.0),   
    up=dict(x=-1, y=0, z=0),
    projection=dict(type='orthographic')
  )
  fig.update_layout(scene=dict(aspectmode='data',
                              xaxis=dict(visible=False),
                              yaxis=dict(visible=False),
                              zaxis=dict(visible=False),camera=camera),
                    margin=dict(l=0,r=0,b=0,t=0),
                    )
  return fig

def _nparray(arr):
  return np.array(stop_gradient(arr),)