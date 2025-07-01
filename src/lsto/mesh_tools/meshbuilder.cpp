#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <vector>
#include <iostream>
#include <array>
#include <map>
#include <unordered_map>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Mesh;
namespace PMP = CGAL::Polygon_mesh_processing;

std::vector<int> get_static_nid(const std::vector<std::array<int, 8>>& cubes) {
    std::unordered_map<int, int> counts;

    // Count occurrences of each unique cube ID
    for (const auto& cube : cubes) {
        for (int id : cube) {
            counts[id]++;
        }
    }

    // Collect variable IDs (those that appear less than 8 times)
    std::vector<int> static_nid;
    for (const auto& entry : counts) {
        if (entry.second != 8) {
            static_nid.push_back(entry.first);
        }
    }
    return static_nid;
}

py::tuple meshbuilder(const Eigen::MatrixXd& coords, const Eigen::MatrixXi& connectivity,
                      double x_grid_size, double y_grid_size, double z_grid_size) {
    int num_vertices = coords.rows();
    int num_faces = connectivity.rows();

    // Create the mesh from the input vertex and connectivity data
    Mesh mesh;

    for (int i = 0; i < num_vertices; ++i) {
        mesh.add_vertex(Point(coords(i, 0), coords(i, 1), coords(i, 2)));
    }

    for (int i = 0; i < num_faces; ++i) {
        std::vector<Mesh::Vertex_index> face_vertices;
        for (int j = 0; j < 3; ++j) {  // Assuming triangular faces
            face_vertices.push_back(Mesh::Vertex_index(connectivity(i, j)));
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
    std::set<int> used_indices;

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

                if ((inside(p000) == CGAL::ON_BOUNDED_SIDE || inside(p000) == CGAL::ON_BOUNDARY) &&
                    (inside(p100) == CGAL::ON_BOUNDED_SIDE || inside(p100) == CGAL::ON_BOUNDARY) &&
                    (inside(p110) == CGAL::ON_BOUNDED_SIDE || inside(p110) == CGAL::ON_BOUNDARY) &&
                    (inside(p010) == CGAL::ON_BOUNDED_SIDE || inside(p010) == CGAL::ON_BOUNDARY) &&
                    (inside(p001) == CGAL::ON_BOUNDED_SIDE || inside(p001) == CGAL::ON_BOUNDARY) &&
                    (inside(p101) == CGAL::ON_BOUNDED_SIDE || inside(p101) == CGAL::ON_BOUNDARY) &&
                    (inside(p111) == CGAL::ON_BOUNDED_SIDE || inside(p111) == CGAL::ON_BOUNDARY) &&
                    (inside(p011) == CGAL::ON_BOUNDED_SIDE || inside(p011) == CGAL::ON_BOUNDARY)) {
                
                //if ((inside(p000) == CGAL::ON_BOUNDED_SIDE || inside(p100) == CGAL::ON_BOUNDED_SIDE || 
                //    inside(p110) == CGAL::ON_BOUNDED_SIDE || inside(p010) == CGAL::ON_BOUNDED_SIDE || 
                //    inside(p001) == CGAL::ON_BOUNDED_SIDE || inside(p101) == CGAL::ON_BOUNDED_SIDE || 
                //    inside(p111) == CGAL::ON_BOUNDED_SIDE || inside(p011) == CGAL::ON_BOUNDED_SIDE)) {
                    
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
                    for (int id : cube) {
                        used_indices.insert(id);
                    }
                }
            }
        }
    }

    std::vector<Point> filtered_grid_points;
    std::map<int, int> old_to_new_index;
    int new_index = 0;

    for (size_t i = 0; i < grid_points.size(); ++i) {
        if (used_indices.count(i)) {
            filtered_grid_points.push_back(grid_points[i]);
            old_to_new_index[i] = new_index++;
        }
    }

    for (auto& cube : cubes) {
        for (int& id : cube) {
            id = old_to_new_index[id];
        }
    }

    Eigen::MatrixXd grid_points_matrix(filtered_grid_points.size(), 3);
    for (size_t i = 0; i < filtered_grid_points.size(); ++i) {
        grid_points_matrix(i, 0) = filtered_grid_points[i].x();
        grid_points_matrix(i, 1) = filtered_grid_points[i].y();
        grid_points_matrix(i, 2) = filtered_grid_points[i].z();
    }

    Eigen::MatrixXi cubes_matrix(cubes.size(), 8);
    for (size_t i = 0; i < cubes.size(); ++i) {
        for (size_t j = 0; j < 8; ++j) {
            cubes_matrix(i, j) = cubes[i][j];
        }
    }

    std::vector<int> static_nid = get_static_nid(cubes);
    Eigen::VectorXi static_nid_vector(static_nid.size());
    for (size_t i = 0; i < static_nid.size(); ++i) {
        static_nid_vector(i) = static_nid[i];
    }

    return py::make_tuple(grid_points_matrix, cubes_matrix, static_nid_vector);
}

PYBIND11_MODULE(meshbuilder, m) {
    m.def("meshbuilder", &meshbuilder, 
          "Process mesh data from Eigen matrices and grid sizes",
          py::arg("coords"), py::arg("connectivity"),
          py::arg("x_grid_size"), py::arg("y_grid_size"), py::arg("z_grid_size"));
}
