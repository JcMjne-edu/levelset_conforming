from lsto.wing_box import *
from lsto.mesh_tools.preprocess import *

f_name_dat='./wing2d_dat/nasasc2-0714.dat'
span=8000.
root_chord=2000.
tip_chord=800.
sweep=20.0
ref_span_elem_scale=0.02
thickness=3.
locs_spar=[0.1,0.413,0.75]
locs_lib=list(np.linspace(0,1,21)[:-1])
wing2d=Wing2D(f_name_dat)
wing3d=Wing3D(wing2d,span,root_chord,tip_chord,sweep,locs_spar,locs_lib,ref_span_elem_scale)
wing3d.export_marc('./marc_input/wing3d_skin.dat',thickness,remove_edge=False)

scale=0.03

trimesh=triMesh(wing3d.connectivity_surface,wing3d.coordinates_surface)
trimesh.norm(scale)
ll=1.8
length_lattice=(ll,ll,ll/3.)
meshbuilder=Build_mesh(trimesh,length_lattice)
meshbuilder.remesh_element()