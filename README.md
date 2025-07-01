# levelset_conforming

## Install libraries

### CGAL

In Ubuntu, use apt-get in the following way:
```
sudo apt-get install libcgal-dev
```
To install Eigen
```
sudo apt install libeigen3-dev
```

To get the demos use
```
sudo apt-get install libcgal-demo
```

```-lgmp``` option may be required when compiled

### Gmsh
In Ubuntu, use apt in the following way:
```
sudo apt update
sudo apt install gmsh
```

## Install petsc

```
git clone -b release https://gitlab.com/petsc/petsc.git
cd petsc

./configure --with-debugging=0 --download-mpich --with-cuda=1 --with-cuda-dir=/usr/local/cuda --download-f2cblaslapack --download-slepc --prefix=$HOME/petsc-gpu

make all
make install

# set environment variables
export PETSC_DIR=$HOME/petsc-install
export PETSC_ARCH=arch-linux-c-opt

# install python bindings
pip install petsc4py slepc4py
```