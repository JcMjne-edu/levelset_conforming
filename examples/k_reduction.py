from lsto.rom.k_reduction_6dof import reduction
import numpy as np

ke3=np.load('./FEM_trg/ke3.npy')
ke4=np.load('./FEM_trg/ke4.npy')
connect3=np.load('./FEM_trg/connect3.npy')
connect4=np.load('./FEM_trg/connect4.npy')
coordinates=np.load('./FEM_trg/coordinates.npy')
nid_edge=np.load('./FEM_trg/nid_edge.npy')
nid_tip=np.load('./FEM_trg/nid_tip.npy')
nnode=coordinates.shape[0]
spc_nid=np.where(coordinates[:,1]==0)[0]
kg_rom,kg,spc_dim=reduction(ke3,ke4,connect3,connect4,nnode,nid_tip,spc_nid)
compliance=np.linalg.inv(kg_rom)
np.save('./FEM_trg/compliance.npy',compliance)