# levelset_conforming

## Install libraries

### TetGen
This project requires the `tetgen` executable (Version 1.6). 
If it is not installed in `/usr/bin/tetgen`, please download and build it from the [official WIAS page](https://wias-berlin.de/software/index.jsp?id=TetGen&lang=1).

### Gmsh

In Ubuntu, use apt in the following way:

```bash
sudo apt update
sudo apt install gmsh
```

### Python dependencies

To bridge C++ and Python, `pybind11` is required:

```bash
pip install pybind11
```

---

## Build C++ Extension

Before running the Python scripts, you need to compile the C++ module (`meshbuilder_aug`). Execute the following command in the terminal:

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    meshbuilder_aug.cpp -o meshbuilder_aug$(python3-config --extension-suffix) \
    -lCGAL -I/usr/include/eigen3
```

> [!IMPORTANT]
> This command generates a shared library file (e.g., `.so` file) that Python can import. Ensure this compilation is successful before attempting to run the main scripts.

## Nastran path specification

The `LSOptimizer` class in `optimizer.py` has a parameter for the Nastran executable path.

If you run the code on WSL, but Nastran is installed on Windows, you can use the following path format:

```python
nastran_path='cmd.exe /c C:/Path/To/Nastran/nast20XXX.exe'
```

## Input parameters

### weightrbf

The `weightrbf` parameter in the `LSOptimizer` constractor is the filter matrix for the RBF-based smoothing of the level set function.