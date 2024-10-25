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
  model.write_bdf(fname,write_header=False,interspersed=False,is_double=False)
  newline='nastran krylov1=-1, krylov3=-6, krylov5=0, tetraar=4000.0\n'
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

def run_nastran_eig(dname,connect,coord,young,poisson,rho,num_mode,
                    nid_spc=None,nastran_path=None):
  fname_bdf=dname+'/nastran_eig.bdf'
  fname_op2=dname+'/nastran_eig.op2'
  nastran_input_eig(fname_bdf,np.asarray(connect),np.asarray(coord),young,poisson,rho,num_mode,nid_spc)
  if nastran_path is None:
    nastran_path='C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'  
  command=f'cmd.exe /c {nastran_path} {fname_bdf} out={dname} old=no news=no > nul 2>&1'
  os.system(command)
  eigvecs,eigvals=nastran_output_eig(fname_op2)
  shutil.move(dname+'/nastran_eig.op2',dname+'/nastran_eig_OOD.op2')
  shutil.move(dname+'/nastran_eig.bdf',dname+'/nastran_eig_OOD.bdf')
  return eigvecs,eigvals