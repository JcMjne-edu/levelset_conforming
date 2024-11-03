import numpy as np
ROUND_COORDINATES=10

class MarcModel_tet_eig:
  def __init__(self,title='tetra_eig',ldname='loadcase1',alloc=100,post_increment=1):
    self.title=title
    self.ldname=ldname
    self.alloc=alloc
    self.post_increment=post_increment
  
  def set_nmode(self,num_modes=6):
    self.num_modes=num_modes
  
  def write(self,path):
    texts=[]
    texts.append(f'TITLE               {self.title},\n')
    texts.append('EXTENDED,\n')
    texts.append(f'SIZING, 0, {self.nelements}, {self.n_nodes}, 0\n')
    texts.append(f'ALLOCATE, {self.alloc},\n')
    texts.append(f'ELEMENTS, 134\n')
    texts.append('VERSION, 13,\n')
    texts.append('TABLE, 0, 0, 2, 1, 1, 0, 0, 1\n')
    texts.append('PROCESSOR, 1, 1, 1, 0\n')
    texts.append('$NO LIST,\n')
    texts.append(f'DYNAMIC, 1, {self.num_modes}, 1, 0, 0, 0, 0, 0, 0\n')
    texts.append('ALL POINTS,\n')
    texts.append('LUMP, 1, 0\n')
    #texts.append('PRINT, 1\n')
    texts.append('NO ECHO, 1, 2, 3\n')
    texts.append(f'END,\n')
    texts.append('SOLVER,\n8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\n')
    texts.append('OPTIMIZE, 11\n')
    self._write_connectivity(texts)
    self._write_coordinates(texts)
    self._write_spc(texts)
    self._write_material(texts)
    self._write_geometry(texts)
    self._write_fixed_disp(texts)
    self._write_loadcase(texts)
    #texts.append('NO PRINT,\n')
    self._write_print_node(texts)
    self._write_post(texts)
    self._write_parameters(texts)
    texts.append('END OPTION,\n')
    self._write_title(texts)
    self._write_loadcase_his(texts)
    self._write_control(texts)
    self._write_modal_shape(texts)
    texts.append('CONTINUE,\n')
    self._write_recover(texts)
    texts.append('CONTINUE,\n')
    with open(path,mode='w') as f:
      f.write(''.join(texts))

  def add_connectivity(self,connectivity):
    self.c_tetra=np.asarray(connectivity)+1
    self.nelements=self.c_tetra.shape[0]
    
  def _write_connectivity(self,texts):
    texts.append('CONNECTIVITY,\n')
    texts.append(f' 0, 0, 1, 0, 1, 1, 0, 0, 0\n')
    order_nelements=int(np.log10(self.nelements))+1
    c_tetra_str=np.array(self.c_tetra).astype(str)
    for i,tetra in enumerate(c_tetra_str):
      text=f' {i+1:{order_nelements}}, 134, '+', '.join(tetra)+'\n'
      texts.append(text)
    
  def add_coordinates(self,coordinates):
    """
    coordinates: (n_nodes,4)
    """
    self.n_nodes=coordinates.shape[0] #(n_nodes,)
    self.coordinates=np.asarray(coordinates)

  def add_spc(self):
    self.spc=np.where(self.coordinates[:,1]==0.0)[0]+1 #(nspc,)

  def _write_spc(self,texts):
    texts.append('DEFINE, NODE, SET, spc_nodes\n')
    for i,spcnode in enumerate(self.spc):
      texts.append(f'{spcnode}, ')
      if i%13==12 and i!=len(self.spc)-1:
        texts.append('c\n')
      elif i==len(self.spc)-1:
        texts.append('\n')

  def _write_fixed_disp(self,texts):
    """
    ------------------------
    """
    texts.append('FIXED DISP,\n\n')
    texts.append('1, 0, 0, 0, 1, 0, apply1\n')
    texts.append('0.0, 0.0, 0.0\n')
    texts.append('0, 0, 0\n')
    texts.append('1, 2, 3\n')
    texts.append('2,\n')
    texts.append('spc_nodes,\n')

  def _write_coordinates(self,texts):
    texts.append('COORDINATES,\n')
    texts.append(f' 3, {self.n_nodes}, 0, 1\n')
    for id,(x,y,z) in enumerate(self.coordinates):
      texts.append(f' {int(id+1)}, {x:.10f}, {y:.10f}, {z:.10f}\n')

  def add_material(self,young,poisson,density):
    self.young=young
    self.poisson=poisson
    self.density=density
  
  def _write_material(self,texts):
    texts.append('ISOTROPIC,\n\n')
    texts.append('1, ELASTIC, ISOTROPIC, 10, 0, 0, 0, material1\n')
    texts.append(f'{self.young}, {self.poisson}, {self.density}, 0.0, 0.0, 0.0, 0.0, 0.0\n')
    texts.append('0, 0, 0, 0, 0, 0, 0, 0\n\n')

  def _write_loadcase(self,texts):
    texts.append('LOADCASE, job1\n')
    texts.append('1,\n')
    texts.append('apply1,\n')
  
  def _write_loadcase_his(self,texts):
    texts.append(f'LOADCASE, {self.ldname}\n')
    texts.append('1,\n')
    texts.append('apply1,\n')

  def _write_modal_shape(self,texts):
    texts.append('MODAL SHAPE,\n')
    texts.append(f'0.0, 0.0, {self.num_modes}, 0, 0, 0.0, 0.0, 0.0\n')

  def _write_recover(self,texts):
    texts.append('RECOVER,\n')
    texts.append(f'1, {self.num_modes}, 0\n')
  
  def _write_geometry(self,texts):
    """
    -------------------
    """
    texts.append('GEOMETRY,\n')
    texts.append('0, 0, 2\n')
    texts.append('1, 9\n')
    texts.append('geom1,\n')
    texts.append('0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n\n')
  
  def _write_print_node(self,texts):
    texts.append('PRINT ELEMENT,\n\n\n\n\n\n') 
  
  def _write_post(self,texts):
    texts.append('POST,\n')
    texts.append(f'2, 16, 17, 0, 0, 19, 20, 0, {self.post_increment}, 0, 0, 0, 0, 0, 0, 0\n')
    texts.append('311, 0\n')
    texts.append('401, 0\n')
  
  def _write_parameters(self,texts):
    texts.append('PARAMETERS,\n')
    texts.append('1.0, 1.0E+9, 1.0E+2, 1.0E+6, 2.5E-1, 5.0E-1, 1.5, -5.0E-1\n')
    texts.append('8.625, 20.0, 1.0E-4, 1.0E-6, 1.0, 1.0E-4\n')
    texts.append('8.314, 2.7315E+2, 5.0E-1, 0.0, 5.67051E-8, 1.438769E-2, 2.9979E+8, 1.0E+30\n')
    texts.append('0.0, 0.0, 1.0E+2, 0.0, 1.0, -2.0, 1.0E+6, 3.0\n')
    texts.append('0.0, 0.0, 1.256637061E-6, 8.854187817E-12, 1.2E+2, 1.0E-3, 1.6E+2, 0.0\n')
    texts.append('3.0, 4.0E-1\n')
  
  def _write_title(self,texts):
    texts.append(f'TITLE, {self.ldname}\n')
  
  def _write_control(self,texts):
    texts.append('CONTROL,\n')
    texts.append('99999, 10, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0\n')
    texts.append('0.001 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n')

class MarcModel_tet_static(MarcModel_tet_eig):
  def __init__(self,title='tetra_static',ldname='loadcase1',alloc=100,post_increment=1):
    self.title=title
    self.ldname=ldname
    self.alloc=alloc
    self.post_increment=post_increment
  
  def write(self,path):
    texts=[]
    texts.append(f'TITLE               {self.title},\n')
    texts.append('EXTENDED,\n')
    texts.append(f'SIZING, 0, {self.nelements}, {self.n_nodes}, 0\n')
    texts.append(f'ALLOCATE, {self.alloc},\n')
    texts.append(f'ELEMENTS, 134\n')
    texts.append('VERSION, 13,\n')
    texts.append('TABLE, 0, 0, 2, 1, 1, 0, 0, 1\n')
    texts.append('PROCESSOR, 1, 1, 1, 0\n')
    texts.append('$NO LIST,\n')
    texts.append('ALL POINTS,\n')
    texts.append('NO ECHO, 1, 2, 3\n')
    texts.append(f'END,\n')
    texts.append('SOLVER,\n8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\n')
    texts.append('OPTIMIZE, 11\n')
    self._write_connectivity(texts)
    self._write_coordinates(texts)
    self._write_spc(texts)
    self._write_single_point_load_node(texts)
    self._write_material(texts)
    self._write_geometry(texts)
    self._write_fixed_disp(texts)
    self._write_single_point_load(texts)
    self._write_loadcase(texts)
    self._write_print_node(texts)
    self._write_post(texts)
    self._write_parameters(texts)
    texts.append('END OPTION,\n')
    self._write_title(texts)
    self._write_loadcase(texts)
    self._write_control(texts)
    self._write_parameters(texts)
    self._write_auto_load(texts)
    self._write_time_step(texts)
    texts.append('CONTINUE,\n')
    with open(path,mode='w') as f:
      f.write(''.join(texts))

  def add_spc(self):
    self.spc=np.where(self.coordinates[:,1]==0.0)[0]+1 #(nspc,)

  def _write_spc(self,texts):
    texts.append('DEFINE, NODE, SET, spc_nodes\n')
    for i,spcnode in enumerate(self.spc):
      texts.append(f'{spcnode}, ')
      if i%13==12 and i!=len(self.spc)-1:
        texts.append('c\n')
      elif i==len(self.spc)-1:
        texts.append('\n')

  def add_single_point_load(self,point_load_nid):
    self.single_point_load_nid=point_load_nid+1

  def _write_single_point_load_node(self,texts):
    texts.append('DEFINE, NODE, SET, point_load_node\n')
    texts.append(f'{self.single_point_load_nid},\n')

  def _write_single_point_load(self,texts):
    texts.append('POINT LOAD,\n\n')
    texts.append('1, 0, 0, 0, 0, 0, apply2\n')
    texts.append('0.0, 0.0, 1.0\n')
    texts.append('0, 0, 0\n')
    texts.append('2,\n')
    texts.append('point_load_node,\n')

  def _write_loadcase(self,texts):
    texts.append('LOADCASE, job1\n')
    texts.append('2,\n')
    texts.append('apply1,\napply2,\n')
  
  def _write_post(self,texts):
    texts.append('POST,\n')
    texts.append(f'0, 16, 17, 0, 0, 19, 20, 0, {self.post_increment}, 0, 0, 0, 0, 0, 0, 0\n')
  
  def _write_parameters(self,texts):
    texts.append('PARAMETERS,\n')
    texts.append('1.0, 1.0E+9, 1.0E+2, 1.0E+6, 2.5E-1, 5.0E-1, 1.5, -5.0E-1\n')
    texts.append('8.625, 20.0, 1.0E-4, 1.0E-6, 1.0, 1.0E-4\n')
    texts.append('8.314, 2.7315E+2, 5.0E-1, 0.0, 5.67051E-8, 1.438769E-2, 2.9979E+8, 1.0E+30\n')
    texts.append('0.0, 0.0, 1.0E+2, 0.0, 1.0, -2.0, 1.0E+6, 3.0\n')
    texts.append('0.0, 0.0, 1.256637061E-6, 8.854187817E-12, 1.2E+2, 1.0E-3, 1.6E+2, 0.0\n')
    texts.append('3.0, 4.0E-1\n')
  
  def _write_title(self,texts):
    texts.append(f'TITLE, {self.ldname}\n')
  
  def _write_control(self,texts):
    texts.append('CONTROL,\n')
    texts.append('99999, 10, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0\n')
    texts.append('0.01 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n')

  def _write_auto_load(self,texts):
    texts.append('AUTO LOAD,\n')
    texts.append('1, 0, 10, 0, 0\n')
  
  def _write_time_step(self,texts):
    texts.append('TIME STEP,\n')
    texts.append('1.0,\n')
  
class Marcmodel_tet_static_MultiplePointLoads(MarcModel_tet_static):
  def write(self,path):
    texts=[]
    texts.append(f'TITLE               {self.title},\n')
    texts.append('EXTENDED,\n')
    texts.append(f'SIZING, 0, {self.nelements}, {self.n_nodes}, 0\n')
    texts.append(f'ALLOCATE, {self.alloc},\n')
    texts.append(f'ELEMENTS, 134\n')
    texts.append('VERSION, 13,\n')
    texts.append('TABLE, 0, 0, 2, 1, 1, 0, 0, 1\n')
    texts.append('PROCESSOR, 1, 1, 1, 0\n')
    texts.append('$NO LIST,\n')
    texts.append('ALL POINTS,\n')
    texts.append('NO ECHO, 1, 2, 3\n')
    texts.append(f'END,\n')
    texts.append('SOLVER,\n8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\n')
    texts.append('OPTIMIZE, 11\n')
    self._write_connectivity(texts)
    self._write_coordinates(texts)
    self._write_spc(texts)
    self._write_node_sets(texts)
    self._write_material(texts)
    self._write_geometry(texts)
    self._write_fixed_disp(texts)
    self._write_point_loads(texts)
    self._write_loadcase(texts)
    self._write_print_node(texts)
    self._write_post(texts)
    self._write_parameters(texts)
    texts.append('END OPTION,\n')
    self._write_title(texts)
    self._write_loadcase(texts)
    self._write_control(texts)
    self._write_parameters(texts)
    self._write_auto_load(texts)
    self._write_time_step(texts)
    texts.append('CONTINUE,\n')
    with open(path,mode='w') as f:
      f.write(''.join(texts))

  def _write_node_sets(self,texts):
    offset=1
    for nid in self.point_load_nids:
      texts.append(f'DEFINE, NODE, SET, apply{nid+offset}_nodes\n{nid},\n')

  def add_point_loads(self,nids,loads):
    """
    nids : int (nnode,)
    loads : float (nnode,3)
    """
    self.point_load_nids=nids+1
    self.point_loads=loads
  
  def _write_point_loads(self,texts):
    offset=1
    texts.append('POINT LOAD,\n\n')
    for nid,load in zip(self.point_load_nids,self.point_loads):
      texts.append(f'1, 0, 0, 0, 0, 0, apply{nid+offset}\n')
      texts.append('{:.10f}, {:.10f}, {:.10f}\n0,0,0\n2,\n'.format(*load))
      texts.append(f'apply{nid+offset}_nodes,\n')

  def _write_loadcase(self,texts):
    offset=1
    texts.append('LOADCASE, job1\n')
    texts.append(f'{offset+self.point_load_nids.shape[0]},\n')
    texts.append('apply1,\n')
    temp=[f'apply{nid+offset},\n' for nid in self.point_load_nids]
    texts.append(''.join(temp))

def marc_from_connect_and_coord_tet_eig(connectivity,coordinates,f_name,young=7e4,poisson=0.3,rho=2.7e-9,num_mode=6):
  model=MarcModel_tet_eig()
  model.add_connectivity(connectivity)
  model.add_coordinates(coordinates)
  model.add_spc()
  model.set_nmode(num_mode)
  model.add_material(young,poisson,rho)
  model.write(f_name)

def marc_from_connect_and_coord_tet_static(connectivity,coordinates,spl_nid,f_name,young=7e4,poisson=0.3,rho=2.7e-9,):
  model=MarcModel_tet_static()
  model.add_connectivity(connectivity)
  model.add_coordinates(coordinates)
  model.add_spc()
  model.add_single_point_load(spl_nid)
  model.add_material(young,poisson,rho)
  model.write(f_name)
  
def marc_static_MultiplePointLoads(connectivity,coordinates,young=7e4,poisson=0.3,rho=2.7e-9,):
  model=Marcmodel_tet_static_MultiplePointLoads()
  model.add_connectivity(connectivity)
  model.add_coordinates(coordinates)
  model.add_spc()
  model.add_material(young,poisson,rho)
  return model