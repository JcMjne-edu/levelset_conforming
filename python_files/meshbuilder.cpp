#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <vector>
#include <iostream>
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Mesh;
namespace PMP = CGAL::Polygon_mesh_processing;

std::vector<int> get_static_nid(const std::vector<std::array<int, 8>>& cubes, const std::vector<Point>& grid_points) {
    std::unordered_map<int, int> counts;

    // Count occurrences of each unique cube ID
    for (const auto& cube : cubes) {
        for (int id : cube) {
            counts[id]++;
        }
    }

    // Collect variable IDs (those that appear 6 times)
    std::vector<int> static_nid;
    for (const auto& entry : counts) {
        if (entry.second != 8) {
            static_nid.push_back(entry.first);
        }
    }
    return static_nid;
}

py::tuple meshbuilder(py::array_t<double> coords_array, py::array_t<int> connectivity_array,
                       double x_grid_size, double y_grid_size, double z_grid_size) {
    // Extract input data from NumPy arrays
    py::buffer_info coords_info = coords_array.request();
    py::buffer_info conn_info = connectivity_array.request();

    double* coords = static_cast<double*>(coords_info.ptr);
    int* connectivity = static_cast<int*>(conn_info.ptr);

    int num_vertices = coords_info.shape[0];
    int num_faces = conn_info.shape[0];

    // Create the mesh from the input vertex and connectivity data
    Mesh mesh;

    for (int i = 0; i < num_vertices; ++i) {
        mesh.add_vertex(Point(coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]));
    }

    for (int i = 0; i < num_faces; ++i) {
        std::vector<Mesh::Vertex_index> face_vertices;
        for (int j = 0; j < 3; ++j) {  // Assuming triangular faces
            face_vertices.push_back(Mesh::Vertex_index(connectivity[3 * i + j]));
        }
        mesh.add_face(face_vertices);
    }

    CGAL::Bbox_3 bbox = PMP::bbox(mesh);
    CGAL::Side_of_triangle_mesh<Mesh, K> inside(mesh);

    std::vector<Point> grid_points;
    std::map<Point, int> point_to_index;
    int index = 0;

    for (double x = bbox.xmin(); x <= bbox.xmax(); x += x_grid_size) {
        for (double y = bbox.ymin(); y <= bbox.ymax(); y += y_grid_size) {
            for (double z = bbox.zmin(); z <= bbox.zmax(); z += z_grid_size) {
                Point p(x, y, z);
                if (inside(p) == CGAL::ON_BOUNDED_SIDE || inside(p) == CGAL::ON_BOUNDARY) {
                    grid_points.push_back(p);
                    point_to_index[p] = index++;
                }
            }
        }
    }

    std::vector<std::array<int, 8>> cubes;
    std::set<int> used_indices;  // Use a set to keep track of used indices

    // Generate cubes
    for (double x = bbox.xmin(); x < bbox.xmax(); x += x_grid_size) {
        for (double y = bbox.ymin(); y < bbox.ymax(); y += y_grid_size) {
            for (double z = bbox.zmin(); z < bbox.zmax(); z += z_grid_size) {
                Point p000(x, y, z);
                Point p100(x + x_grid_size, y, z);
                Point p010(x, y + y_grid_size, z);
                Point p001(x, y, z + z_grid_size);
                Point p101(x + x_grid_size, y, z + z_grid_size);
                Point p011(x, y + y_grid_size, z + z_grid_size);
                Point p110(x + x_grid_size, y + y_grid_size, z);
                Point p111(x + x_grid_size, y + y_grid_size, z + z_grid_size);

                // Check if all corners of the cube are inside the mesh
                if ((inside(p000) == CGAL::ON_BOUNDED_SIDE || inside(p000) == CGAL::ON_BOUNDARY)&&
                    (inside(p100) == CGAL::ON_BOUNDED_SIDE || inside(p100) == CGAL::ON_BOUNDARY)&&
                    (inside(p110) == CGAL::ON_BOUNDED_SIDE || inside(p110) == CGAL::ON_BOUNDARY)&&
                    (inside(p010) == CGAL::ON_BOUNDED_SIDE || inside(p010) == CGAL::ON_BOUNDARY)&&
                    (inside(p001) == CGAL::ON_BOUNDED_SIDE || inside(p001) == CGAL::ON_BOUNDARY)&&
                    (inside(p101) == CGAL::ON_BOUNDED_SIDE || inside(p101) == CGAL::ON_BOUNDARY)&&
                    (inside(p111) == CGAL::ON_BOUNDED_SIDE || inside(p111) == CGAL::ON_BOUNDARY)&&
                    (inside(p011) == CGAL::ON_BOUNDED_SIDE || inside(p011) == CGAL::ON_BOUNDARY)) {
                    
                    std::array<int, 8> cube = {
                        point_to_index[p000],
                        point_to_index[p100],
                        point_to_index[p110],
                        point_to_index[p010],
                        point_to_index[p001],
                        point_to_index[p101],
                        point_to_index[p111],
                        point_to_index[p011]
                    };
                    cubes.push_back(cube);
                    // Mark these indices as used
                    for (int id : cube) {
                        used_indices.insert(id);
                    }
                }
            }
        }
    }

    std::vector<Point> filtered_grid_points;
    std::map<int, int> old_to_new_index; // Maps old index to new index
    int new_index = 0;

    for (size_t i = 0; i < grid_points.size(); ++i) {
        if (used_indices.count(i)) {
            filtered_grid_points.push_back(grid_points[i]);
            old_to_new_index[i] = new_index++;
        }
    }

    // Update cubes to use the new indices
    for (auto& cube : cubes) {
        for (int& id : cube) {
            id = old_to_new_index[id]; // Update to new index
        }
    }

    // Get static NIDs
    std::vector<int> static_nid = get_static_nid(cubes, filtered_grid_points);

    // Convert grid points to a NumPy array
    py::array_t<double> grid_points_array = py::array_t<double>(filtered_grid_points.size() * 3);
    auto ptr = grid_points_array.mutable_unchecked<1>();
    for (size_t i = 0; i < filtered_grid_points.size(); ++i) {
        ptr(3 * i) = filtered_grid_points[i].x();
        ptr(3 * i + 1) = filtered_grid_points[i].y();
        ptr(3 * i + 2) = filtered_grid_points[i].z();
    }

    // Convert grid points to a NumPy array
    //py::array_t<double> grid_points_array = py::array_t<double>(grid_points.size() * 3);
    //auto ptr = grid_points_array.mutable_unchecked<1>();
    //for (size_t i = 0; i < grid_points.size(); ++i) {
    //    ptr(3 * i) = grid_points[i].x();
    //    ptr(3 * i + 1) = grid_points[i].y();
    //    ptr(3 * i + 2) = grid_points[i].z();
    //}

    // Convert cubes to a NumPy array
    py::array_t<int> cubes_array = py::array_t<int>(cubes.size() * 8);
    auto cube_ptr = cubes_array.mutable_unchecked<1>();
    for (size_t i = 0; i < cubes.size(); ++i) {
        for (size_t j = 0; j < 8; ++j) {
            cube_ptr(8 * i + j) = cubes[i][j];
        }
    }

    // Convert static NIDs to a NumPy array
    py::array_t<int> static_nid_array = py::array_t<int>(static_nid.size());
    auto static_nid_ptr = static_nid_array.mutable_unchecked<1>();
    for (size_t i = 0; i < static_nid.size(); ++i) {
        static_nid_ptr(i) = static_nid[i];
    }


    // Return both arrays
    return py::make_tuple(grid_points_array, cubes_array, static_nid_array);
}


// Pybind11 module
PYBIND11_MODULE(meshbuilder, m) {
    m.def("meshbuilder", &meshbuilder, 
          "Process mesh data from NumPy arrays and grid sizes",
          py::arg("coords_array"), py::arg("connectivity_array"),
          py::arg("x_grid_size"), py::arg("y_grid_size"), py::arg("z_grid_size"));
}
