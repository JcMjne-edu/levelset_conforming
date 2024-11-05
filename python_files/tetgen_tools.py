import numpy as np

def make_poly(fname,coord,connect,hole=None):
    """
    coord : np.ndarray (n,3)
    connect : np.ndarray (m,3)
    hole : np.ndarray (l,3)
    fname : str
    """
    print('used poly')
    with open(fname, 'w') as f:
        # Write node information
        npoints = coord.shape[0]
        f.write(f'{npoints}  3  0  0\n')
        np.savetxt(f, np.column_stack((np.arange(1, npoints + 1), coord)),
                   fmt='%d %.15g %.15g %.15g')

        # Write face information
        nfaces = connect.shape[0]
        f.write(f'{nfaces}  0\n')
        faces = np.column_stack((np.ones(nfaces, dtype=int) * 1, 
                                 np.ones(nfaces, dtype=int) * 3,
                                 connect + 1))
        np.savetxt(f, faces, fmt='%d\n%d %d %d %d')

        # Write hole information
        if hole is None:
            f.write('0\n')
        else:
            nholes = hole.shape[0]
            f.write(f'{nholes}\n')
            hole_data = np.column_stack((np.arange(1, nholes + 1), hole))
            np.savetxt(f, hole_data, fmt='%d %.15g %.15g %.15g')
        f.write('0')

def postprocess_tetgen(dir,name,n):
  fname_node=f'{dir}/{name}.{n}.node'
  fname_ele=f'{dir}/{name}.{n}.ele'
  # read node
  nodes=np.loadtxt(fname_node, skiprows=1, usecols=(1, 2, 3))
  #read ele
  ele=np.loadtxt(fname_ele, skiprows=1, usecols=(1, 2, 3, 4)).astype(int) - 1
  return nodes,ele#,face
