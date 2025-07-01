import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
from jax import config
config.update("jax_enable_x64", True)
from importlib import reload
from jax import grad
from lsto.wing_box import *
import lsto.fem.tetra4_fem
import matplotlib.pyplot as plt
from lsto.aeroelastic_scaling import *
import lsto.levelset_conforming_cgal
import lsto.levelset_conforming_penetrate
import lsto.mesh_tools.mapping_surfacenode_full
import lsto.fem.fem_tools
import lsto.fem.tetra4_fem
import lsto.rom.guyan_reduction
import lsto.tetgen_tools
import lsto.mesh_tools.mesh_utility
import lsto.mesh_tools.mesh_postprocess_jax
import lsto.mesh_tools.mesh_postprocess_np
import lsto.nastran_tools
import lsto.custom_eig_external_general
import lsto.cgal_tools
reload(lsto.levelset_conforming_cgal)
reload(lsto.levelset_conforming_penetrate)
reload(lsto.fem.fem_tools)
reload(lsto.fem.tetra4_fem)
reload(lsto.rom.guyan_reduction)
reload(lsto.tetgen_tools)
reload(lsto.mesh_tools.mesh_utility)
reload(lsto.mesh_tools.mesh_postprocess_jax)
reload(lsto.mesh_tools.mesh_postprocess_np)
reload(lsto.nastran_tools)
reload(lsto.custom_eig_external_general)
reload(lsto.mesh_tools.mapping_surfacenode_full)
reload(lsto.cgal_tools)
from lsto.custom_eig_external_general import *
from lsto.nastran_tools import *
from lsto.mesh_tools.mesh_postprocess_jax import *
from lsto.mesh_tools.mesh_postprocess_np import *
from lsto.tetgen_tools import *
from lsto.fem.tetra4_fem import *
from  lsto.fem.fem_tools import *
from lsto.rom.guyan_reduction import *
from lsto.mesh_tools.mesh_utility import *
from lsto.levelset_conforming_cgal import *
from lsto.levelset_conforming_penetrate import *
from lsto.mesh_tools.mapping_surfacenode_full import *
from lsto.cgal_tools import *

