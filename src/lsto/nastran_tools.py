import os
from pyNastran.bdf.bdf import BDF,CaseControlDeck
from pyNastran.op2.op2 import OP2
import numpy as np
import shutil

def nastran_input_eig(fname,connect,coord,young,poisson,rho,num_mode,nid_spc=None):
  model=BDF(debug=None)
  model.sol=103
  cc=CaseControlDeck(['TITLE=eigenan value alysis','ECHO=NONE','SPC=1',
                      'DISP(PLOT,NORPRINT)=ALL','AUTOSPC(NOPRINT)=YES',
                      'METHOD(STRUCTURE)=100'])
  model.case_control_deck=cc
  model.add_param('POST',-1)
  model.add_eigrl(sid=100,nd=num_mode)
  model.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)
  model.add_psolid(pid=1,mid=1)
  
  for i in range(len(coord)):
    model.add_grid(nid=i+1,xyz=coord[i])
  for i in range(len(connect)):
    model.add_ctetra(eid=i+1,pid=1,nids=connect[i]+1)

  if nid_spc is None:
    nid_spc=np.where(coord[:,1]==0.0)[0]+1
  #print(nid_spc)
  model.add_spc1(conid=1,components='123456',nodes=nid_spc)
  model.write_bdf(fname,write_header=False,interspersed=False,is_double=True)
  newline='nastran krylov1=-1, krylov3=-6, krylov5=0, tetraar=400000.0\n'
  with open(fname, 'r') as file:
    lines = file.readlines()
  lines.insert(0, newline)
  with open(fname, 'w') as file:
    file.writelines(lines)

def nastran_output_eig(fname):
  model=OP2(debug=None)
  model.read_op2(fname)
  eigvecs=model.eigenvectors[1].data
  eigvecs=eigvecs[:,:,:3].reshape(eigvecs.shape[0],-1).T
  eigvals=np.array(model.eigenvectors[1].eigns)
  return eigvecs,eigvals

def run_nastran_eig(connect,coord,nastran_path,double=False):
  fname_bdf='./nastran/nastran_eig.bdf'
  fname_op2='./nastran/nastran_eig.op2'
  write_data(connect,coord,double)
  command=f'{nastran_path} {fname_bdf} out=./nastran old=no news=no > nul 2>&1'
  os.system(command)
  eigvecs,eigvals=nastran_output_eig(fname_op2)
  shutil.move('./nastran/nastran_eig.op2','./nastran/nastran_eig_OOD.op2')
  #shutil.move('./nastran/nastran_eig.bdf','./nastran/nastran_eig_OOD.bdf')
  return eigvecs,eigvals

def write_base_nastran(num_mode,young,poisson,rho):
  fname='./nastran/nastran_eig.bdf'
  model=BDF(debug=None)
  model.sol=103
  cc=CaseControlDeck(['TITLE=eigen value alysis','ECHO=NONE','SPC=1',
                      'DISP(PLOT,NORPRINT)=ALL','AUTOSPC(NOPRINT)=YES',
                      'METHOD(STRUCTURE)=100'])
  model.case_control_deck=cc
  model.add_param('POST',-1)
  model.add_eigrl(sid=100,nd=num_mode)
  model.add_mat1(mid=1,E=young,G=None,nu=poisson,rho=rho)
  model.add_psolid(pid=1,mid=1)
  model.add_include_file('./nastran/data.bdf')
  model.write_bdf(fname,write_header=False,interspersed=False,)
  newline='nastran krylov1=-1, tetraar=40000.0\n'
  with open(fname, 'r') as file:
    lines = file.readlines()
  lines.insert(0, newline)
  with open(fname, 'w') as file:
    file.writelines(lines)

def write_data(connects,coords,double=False):
  npoints=coords.shape[0]
  nelems=connects.shape[0]
  coords=coords-coords.min(axis=0)
  n_int=int(np.log10(np.abs(coords).max()))
  if double:
    n=14-n_int
    fmt_grid = f'GRID*   %16d                %16.{n}f%16.{n}f\n*       %16.{n}f'
  else:
    n=6-n_int
    fmt_grid = f'GRID    %8d        %8.{n}f%8.{n}f%8.{n}f'
  fmt_elem = 'CTETRA  %8d       1%8d%8d%8d%8d'
  spc=np.where(coords[:,1]==0.0)[0]+1
  n_spc=spc.shape[0]
  n_lines=np.ceil((n_spc-6)/8).astype(int)+1
  line_spc1_base='SPC1           1  123456'
  if n_lines==1:
    line_spc=line_spc1_base+('{:8}'*n_spc).format(*spc)
  elif n_lines==2:  
    line_spc1=line_spc1_base+('{:8}'*6).format(*spc[:6])+'\n'
    line_spc2=' '*8+(('{:8}'*(n_spc-6)).format(*spc[6:]))
    line_spc=line_spc1+line_spc2
  else:
    line_spc1=line_spc1_base+('{:8}'*6).format(*spc[:6])+'\n'
    n_line_middle=n_lines-2
    line_spc2=[' '*8+(('{:8}'*8).format(*spc[6+i*8:14+i*8])) for i in range(n_line_middle)]
    line_spc2='\n'.join(line_spc2)+'\n'
    line_spc3=' '*8+(('{:8}'*(n_spc-6-n_line_middle*8)).format(*spc[6+n_line_middle*8:]))
    line_spc=line_spc1+line_spc2+line_spc3
  with open('./nastran/data.bdf','w') as f:
    np.savetxt(f,np.column_stack((np.arange(1,npoints+1),coords)),fmt=fmt_grid)
    np.savetxt(f,np.column_stack((np.arange(1,nelems+1),connects+1)),fmt=fmt_elem)
    f.write(line_spc)
